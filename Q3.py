import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass, field
from scipy import stats
from scipy.optimize import minimize_scalar
import statsmodels.api as sm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# ==========================================
# 全局配置
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
warnings.filterwarnings('ignore')

# 医学常数
FF_THRESHOLD = 0.04
MIN_GA_WEEKS = 10
MAX_GA_WEEKS = 28

# 颜色方案
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#28A745',
    'warning': '#FFC107',
    'danger': '#DC3545',
    'info': '#17A2B8',
}

# 决策树参数
TREE_PARAMS = {
    'min_samples_leaf': 20,
    'max_depth': 4,
    'min_samples_split': 40,
    'min_variance_decrease': 0.0001,
    'lambda_risk': 0.80
}

# 特征优先级顺序（强制按此顺序分裂）
# BMI最重要，其次是Weight，然后是Age和Height
FEATURE_PRIORITY = ['BMI', 'Weight', 'Age', 'Height']


# ==========================================
# 数据类定义
# ==========================================
@dataclass
class SplitResult:
    """分裂结果"""
    feature: str
    threshold: float
    score: float
    left_indices: np.ndarray
    right_indices: np.ndarray
    left_optimal_week: float
    right_optimal_week: float
    left_pass_rate: float
    right_pass_rate: float


@dataclass
class TreeNode:
    """决策树节点"""
    node_id: int
    depth: int
    indices: np.ndarray

    sample_size: int = 0
    pass_rate: float = 0.0
    optimal_week: float = 12.0
    compliance_probability: float = 0.5
    risk_score: float = 0.0

    is_leaf: bool = True
    split_feature: str = None
    split_threshold: float = None

    left_child: 'TreeNode' = None
    right_child: 'TreeNode' = None

    path_conditions: List[str] = field(default_factory=list)
    ci_lower: float = 10.0
    ci_upper: float = 14.0


@dataclass
class LeafNodeResult:
    """叶节点最终结果"""
    leaf_id: int
    path_description: str
    conditions: List[str]
    bmi_condition: str
    weight_condition: str
    age_condition: str
    height_condition: str
    sample_size: int
    pass_rate: float
    optimal_week: float
    compliance_probability: float
    risk_score: float
    ci_lower: float
    ci_upper: float
    recommendation: str


# ==========================================
# 数据预处理
# ==========================================
class NIPTDataProcessor:
    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.imputer = IterativeImputer(max_iter=20, random_state=2025)

    def _parse_gestational_age(self, ga_str) -> float:
        if pd.isna(ga_str):
            return np.nan
        try:
            ga_str = str(ga_str).lower().strip().replace('d', '')
            if 'w' in ga_str:
                parts = ga_str.split('w')
                weeks = float(parts[0])
                days = float(parts[1].replace('+', '').strip()) if len(parts) > 1 and parts[1].replace('+',
                                                                                                       '').strip() else 0
                return weeks + days / 7.0
            return float(ga_str)
        except:
            return np.nan

    def process_dataset(self, df_raw: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        df = df_raw.copy()

        if '检测孕周' in df.columns:
            df['GA_numeric'] = df['检测孕周'].apply(self._parse_gestational_age)

        for col in ['年龄', '身高', '体重']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        numeric_cols = [c for c in ['年龄', '身高', '体重'] if c in df.columns]
        if numeric_cols:
            df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])

        if '身高' in df.columns and df['身高'].median() < 3:
            df['身高'] = df['身高'] * 100

        if '身高' in df.columns and '体重' in df.columns:
            df['BMI_calc'] = df['体重'] / ((df['身高'] / 100) ** 2)
            df.loc[(df['BMI_calc'] < 15) | (df['BMI_calc'] > 60), 'BMI_calc'] = np.nan

        if 'Y染色体浓度' in df.columns:
            df['FF_Pass'] = (df['Y染色体浓度'] >= FF_THRESHOLD).astype(int)

        df['Age'] = df['年龄']
        df['Height'] = df['身高']
        df['Weight'] = df['体重']
        df['BMI'] = df['BMI_calc']

        print(f"   📊 {dataset_name}:  n={len(df)}")
        return df

    def load_and_process(self):
        print("=" * 70)
        print("📂 NIPT Data Loading and Preprocessing")
        print("=" * 70)
        try:
            df_male = pd.read_excel(self.excel_path, sheet_name='男胎检测数据')
            df_female = pd.read_excel(self.excel_path, sheet_name='女胎检测数据')
            df_male_processed = self.process_dataset(df_male, 'Male Fetus')
            df_female_processed = self.process_dataset(df_female, 'Female Fetus')
            return df_male_processed, df_female_processed
        except Exception as e:
            print(f"❌ Error:  {e}")
            return None, None


# ==========================================
# 最优检测时点计算器 (关键修改)
# ==========================================
class OptimalTimingCalculator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.lambda_risk = TREE_PARAMS['lambda_risk']

        # 预先拟合全局回归模型
        self._fit_global_model()

    def _fit_global_model(self):
        """拟合全局回归模型：Y浓度 ~ GA + BMI"""
        df_valid = self.df.dropna(subset=['GA_numeric', 'BMI', 'Y染色体浓度'])
        X = df_valid[['GA_numeric', 'BMI']]
        y = df_valid['Y染色体浓度']
        X_sm = sm.add_constant(X)
        self.global_model = sm.OLS(y, X_sm).fit()
        self.residual_std = np.sqrt(self.global_model.mse_resid)

        print(f"   Global Model:  Y = {self.global_model.params['const']:.4f} + "
              f"{self.global_model.params['GA_numeric']:.4f}*GA + "
              f"{self.global_model.params['BMI']:.4f}*BMI")
        print(f"   Residual Std: {self.residual_std:.4f}")

    def _risk_function(self, t: float) -> float:
        """风险函数 - 平滑的S型曲线"""

        def sigmoid(x, k=1.0):
            return 1 / (1 + np.exp(-k * x))

        # 三段式风险：<12周低风险，12-20周中风险，>20周高风险
        risk = 0.05 + 0.25 * sigmoid(t - 12, 0.8) + 0.70 * sigmoid(t - 20, 0.5)
        return risk

    def _predict_concentration(self, ga: float, bmi: float) -> Tuple[float, float]:
        """预测给定GA和BMI下的Y浓度及其不确定性"""
        predicted = (self.global_model.params['const'] +
                     self.global_model.params['GA_numeric'] * ga +
                     self.global_model.params['BMI'] * bmi)
        return predicted, self.residual_std

    def _estimate_pass_probability(self, indices: np.ndarray, t: float) -> float:
        """
        估计在孕周t时的达标概率

        使用回归模型预测，考虑个体差异
        """
        df_group = self.df.iloc[indices]

        if len(df_group) < 5:
            return 0.5

        mean_bmi = df_group['BMI'].mean()
        predicted_y, std = self._predict_concentration(t, mean_bmi)

        # P(Y >= 0. 04) = 1 - Φ((0.04 - μ) / σ)
        prob = 1 - stats.norm.cdf(FF_THRESHOLD, loc=predicted_y, scale=std)

        return float(np.clip(prob, 0.01, 0.99))

    def _find_threshold_week(self, indices: np.ndarray, target_prob: float = 0.85) -> float:
        """
        找到达到目标达标概率所需的最小孕周

        这是核心改进：不是优化复杂的目标函数，而是直接找
        "使达标概率达到target_prob的最早时间点"
        """
        df_group = self.df.iloc[indices]
        mean_bmi = df_group['BMI'].mean()

        # 二分搜索找到达标概率=target_prob的孕周
        low, high = 10.0, 25.0

        while high - low > 0.1:
            mid = (low + high) / 2
            prob = self._estimate_pass_probability(indices, mid)

            if prob >= target_prob:
                high = mid  # 可以更早
            else:
                low = mid  # 需要更晚

        return round((low + high) / 2, 1)

    def find_optimal_timing(self, indices: np.ndarray) -> Tuple[float, float, float, Tuple[float, float]]:
        """
        找到最优检测时点

        改进策略：
        1. 首先找到达标概率>=85%的最早时间点
        2. 然后考虑风险调整
        3. 确保结果在合理范围内
        """
        if len(indices) < 5:
            return 12.0, 0.5, self._risk_function(12.0), (11.0, 13.0)

        df_group = self.df.iloc[indices]
        mean_bmi = df_group['BMI'].mean()
        current_pass_rate = (df_group['Y染色体浓度'] >= FF_THRESHOLD).mean()

        # 策略1：找到85%达标概率的最早时间点
        threshold_week_85 = self._find_threshold_week(indices, target_prob=0.85)

        # 策略2：找到90%达标概率的时间点（更保守）
        threshold_week_90 = self._find_threshold_week(indices, target_prob=0.90)

        # 策略3：基于当前达标率调整
        # 如果当前达标率已经很高，可以更早检测
        if current_pass_rate >= 0.90:
            # 高达标率组：找到80%概率的时间点
            base_week = self._find_threshold_week(indices, target_prob=0.80)
        elif current_pass_rate >= 0.80:
            # 中等达标率组：使用85%概率时间点
            base_week = threshold_week_85
        else:
            # 低达标率组：使用90%概率时间点
            base_week = threshold_week_90

        # 策略4：根据BMI进行微调
        # 高BMI需要额外延迟
        bmi_adjustment = 0
        if mean_bmi >= 35:
            bmi_adjustment = 2.0
        elif mean_bmi >= 30:
            bmi_adjustment = 1.0
        elif mean_bmi >= 28:
            bmi_adjustment = 0.5

        optimal_week = base_week + bmi_adjustment

        optimal_week = np.clip(optimal_week, 10.0, 28.0)
        optimal_week = round(optimal_week, 1)

        # 计算该时点的达标概率和风险
        compliance_prob = self._estimate_pass_probability(indices, optimal_week)
        risk_score = self._risk_function(optimal_week)

        # 置信区间
        se = 1.0 / np.sqrt(len(indices))
        ci = (optimal_week - 1.96 * se, optimal_week + 1.96 * se)

        return optimal_week, compliance_prob, risk_score, ci

# ==========================================
# 层次化分裂优化器（按特征优先级）
# ==========================================
class HierarchicalSplitFinder:
    """
    层次化分裂搜索器

    核心思想：
    - 第1层强制使用BMI（医学上最重要的因素）
    - 第2层优先使用Weight（与BMI相关但提供额外信息）
    - 第3层使用Age
    - 第4层使用Height

    每层在指定特征上找最优阈值
    """

    def __init__(self, df: pd.DataFrame, timing_calculator: OptimalTimingCalculator):
        self.df = df
        self.timing_calc = timing_calculator

    def _evaluate_split(self, indices: np.ndarray, feature: str, threshold: float) -> Optional[SplitResult]:
        """评估单个分裂点"""
        df_subset = self.df.iloc[indices]
        feature_vals = df_subset[feature].values

        # 处理NaN
        valid_mask = ~np.isnan(feature_vals)
        if valid_mask.sum() < TREE_PARAMS['min_samples_split']:
            return None

        left_mask = (feature_vals < threshold) & valid_mask
        right_mask = (feature_vals >= threshold) & valid_mask

        left_idx = indices[left_mask]
        right_idx = indices[right_mask]

        if len(left_idx) < TREE_PARAMS['min_samples_leaf'] or len(right_idx) < TREE_PARAMS['min_samples_leaf']:
            return None

        # 计算子节点最优时点和达标率
        left_week, _, _, _ = self.timing_calc.find_optimal_timing(left_idx)
        right_week, _, _, _ = self.timing_calc.find_optimal_timing(right_idx)

        left_pass = (self.df.iloc[left_idx]['Y染色体浓度'] >= FF_THRESHOLD).mean()
        right_pass = (self.df.iloc[right_idx]['Y染色体浓度'] >= FF_THRESHOLD).mean()

        # 评分：最优时点差异 + 达标率差异
        time_diff = abs(left_week - right_week)
        pass_diff = abs(left_pass - right_pass)

        # 综合得分
        score = time_diff * 1.0 + pass_diff * 5.0

        return SplitResult(
            feature=feature,
            threshold=threshold,
            score=score,
            left_indices=left_idx,
            right_indices=right_idx,
            left_optimal_week=left_week,
            right_optimal_week=right_week,
            left_pass_rate=left_pass,
            right_pass_rate=right_pass
        )

    def find_best_split_for_feature(self, indices: np.ndarray, feature: str) -> Optional[SplitResult]:
        """在指定特征上找最优分裂点"""
        if len(indices) < TREE_PARAMS['min_samples_split']:
            return None

        df_subset = self.df.iloc[indices]
        feature_vals = df_subset[feature].dropna().values

        if len(feature_vals) < TREE_PARAMS['min_samples_split']:
            return None

        best_split = None
        best_score = -np.inf

        # 生成候选阈值
        percentiles = list(range(10, 91, 5))
        candidate_thresholds = np.percentile(feature_vals, percentiles)
        candidate_thresholds = np.unique(np.round(candidate_thresholds, 1))

        for threshold in candidate_thresholds:
            split_result = self._evaluate_split(indices, feature, threshold)

            if split_result is not None and split_result.score > best_score:
                best_score = split_result.score
                best_split = split_result

        return best_split

    def find_best_split(self, indices: np.ndarray, depth: int,
                        used_features: set = None) -> Optional[SplitResult]:
        """
        根据深度和优先级找最佳分裂

        策略：
        - depth=0:  强制使用BMI
        - depth=1: 优先Weight，如果不行用Age
        - depth=2: 优先Age，如果不行用Height
        - depth=3: 使用Height
        """
        if used_features is None:
            used_features = set()

        # 根据深度确定要尝试的特征顺序
        if depth == 0:
            # 第一层强制BMI
            features_to_try = ['BMI']
        elif depth == 1:
            # 第二层优先Weight
            features_to_try = ['Weight', 'Age', 'Height']
        elif depth == 2:
            # 第三层优先Age
            features_to_try = ['Age', 'Height', 'Weight']
        else:
            # 第四层优先Height
            features_to_try = ['Height', 'Age', 'Weight']

        # 移除已使用的特征（可选，这里允许重复使用）
        # features_to_try = [f for f in features_to_try if f not in used_features]

        for feature in features_to_try:
            split = self.find_best_split_for_feature(indices, feature)
            if split is not None:
                return split

        return None


# ==========================================
# 层次化决策树
# ==========================================
class HierarchicalDecisionTree:
    """
    层次化决策树

    特点：
    - BMI作为第一层分裂特征（符合医学逻辑）
    - 后续层按Weight > Age > Height优先级
    - 每个特征可以在不同深度多次使用（不同阈值）
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.dropna(subset=['BMI', 'GA_numeric', 'Y染色体浓度']).reset_index(drop=True)
        self.timing_calc = OptimalTimingCalculator(self.df)
        self.split_finder = HierarchicalSplitFinder(self.df, self.timing_calc)
        self.root = None
        self.leaf_nodes = []
        self.node_counter = 0
        self.split_history = []

    def _create_node(self, indices: np.ndarray, depth: int, path_conditions: List[str]) -> TreeNode:
        node = TreeNode(
            node_id=self.node_counter,
            depth=depth,
            indices=indices,
            path_conditions=path_conditions.copy()
        )
        self.node_counter += 1

        df_node = self.df.iloc[indices]
        node.sample_size = len(indices)
        node.pass_rate = (df_node['Y染色体浓度'] >= FF_THRESHOLD).mean()

        opt_week, prob, risk, ci = self.timing_calc.find_optimal_timing(indices)
        node.optimal_week = opt_week
        node.compliance_probability = prob
        node.risk_score = risk
        node.ci_lower, node.ci_upper = ci

        return node

    def _get_used_features(self, path_conditions: List[str]) -> set:
        """从路径条件中提取已使用的特征"""
        used = set()
        for cond in path_conditions:
            for f in FEATURE_PRIORITY:
                if f in cond:
                    used.add(f)
                    break
        return used

    def _build_tree_recursive(self, node: TreeNode) -> TreeNode:
        """递归构建决策树"""
        if node.depth >= TREE_PARAMS['max_depth']:
            node.is_leaf = True
            return node

        if node.sample_size < TREE_PARAMS['min_samples_split']:
            node.is_leaf = True
            return node

        used_features = self._get_used_features(node.path_conditions)
        best_split = self.split_finder.find_best_split(node.indices, node.depth, used_features)

        if best_split is None:
            node.is_leaf = True
            return node

        node.is_leaf = False
        node.split_feature = best_split.feature
        node.split_threshold = best_split.threshold

        self.split_history.append({
            'node_id': node.node_id,
            'depth': node.depth,
            'feature': best_split.feature,
            'threshold': best_split.threshold,
            'score': best_split.score,
            'left_n': len(best_split.left_indices),
            'right_n': len(best_split.right_indices),
            'left_week': best_split.left_optimal_week,
            'right_week': best_split.right_optimal_week,
            'left_pass': best_split.left_pass_rate,
            'right_pass': best_split.right_pass_rate
        })

        left_cond = node.path_conditions + [f"{best_split.feature} < {best_split.threshold:.1f}"]
        right_cond = node.path_conditions + [f"{best_split.feature} >= {best_split.threshold:.1f}"]

        node.left_child = self._create_node(best_split.left_indices, node.depth + 1, left_cond)
        node.right_child = self._create_node(best_split.right_indices, node.depth + 1, right_cond)

        self._build_tree_recursive(node.left_child)
        self._build_tree_recursive(node.right_child)

        return node

    def fit(self):
        print("\n" + "=" * 70)
        print("📊 Building Hierarchical Decision Tree")
        print("   (BMI as primary split, then Weight > Age > Height)")
        print("=" * 70)

        print(f"\n【Parameters】")
        print(f"   Max Depth: {TREE_PARAMS['max_depth']}")
        print(f"   Min Samples Leaf: {TREE_PARAMS['min_samples_leaf']}")
        print(f"   Feature Priority: {FEATURE_PRIORITY}")

        all_indices = np.arange(len(self.df))
        self.root = self._create_node(all_indices, depth=0, path_conditions=[])

        print(f"\n【Root Node】")
        print(f"   Samples: {self.root.sample_size}")
        print(f"   Pass Rate: {self.root.pass_rate:.1%}")
        print(f"   Global Optimal Week: {self.root.optimal_week:.1f}")

        print(f"\n【Building Tree... 】")
        self._build_tree_recursive(self.root)

        self._collect_leaf_nodes(self.root)

        print(f"\n【Complete】")
        print(f"   Total Nodes: {self.node_counter}")
        print(f"   Leaf Nodes: {len(self.leaf_nodes)}")

        return self

    def _collect_leaf_nodes(self, node: TreeNode):
        if node.is_leaf:
            self.leaf_nodes.append(node)
        else:
            if node.left_child:
                self._collect_leaf_nodes(node.left_child)
            if node.right_child:
                self._collect_leaf_nodes(node.right_child)

    def print_split_history(self):
        print("\n" + "=" * 70)
        print("📊 Split History (Optimized Thresholds)")
        print("=" * 70)

        print("\n" + "-" * 120)
        print(f"{'Depth':<6} {'Feature':<10} {'Threshold':<12} {'Score':<10} "
              f"{'Left n':<8} {'Right n':<8} {'L-Week':<10} {'R-Week':<10} {'L-Pass%':<10} {'R-Pass%':<10}")
        print("-" * 120)

        for split in self.split_history:
            print(f"{split['depth']:<6} {split['feature']:<10} {split['threshold']:<12.1f} "
                  f"{split['score']:<10.3f} {split['left_n']:<8} {split['right_n']:<8} "
                  f"{split['left_week']:<10.1f} {split['right_week']:<10.1f} "
                  f"{split['left_pass'] * 100:<10.1f} {split['right_pass'] * 100:<10.1f}")
        print("-" * 120)

        # 汇总阈值
        print("\n【Discovered Optimal Thresholds】")
        feature_thresholds = {}
        for split in self.split_history:
            feat = split['feature']
            if feat not in feature_thresholds:
                feature_thresholds[feat] = []
            feature_thresholds[feat].append(round(split['threshold'], 1))

        for feat in FEATURE_PRIORITY:
            if feat in feature_thresholds:
                thresholds = sorted(set(feature_thresholds[feat]))
                print(f"   {feat}:  {thresholds}")

    def print_tree_structure(self, node: TreeNode = None, indent: str = ""):
        if node is None:
            node = self.root
            print("\n" + "=" * 70)
            print("📊 Decision Tree Structure")
            print("=" * 70)

        if node.is_leaf:
            print(f"{indent}🍂 Leaf[{node.node_id}]:  n={node.sample_size}, "
                  f"Week={node.optimal_week:.1f}, Pass={node.pass_rate:.1%}")
        else:
            print(f"{indent}🔀 [{node.split_feature} < {node.split_threshold:.1f}]")
            print(f"{indent}   ├── Yes (< {node.split_threshold:.1f}):")
            self.print_tree_structure(node.left_child, indent + "   │   ")
            print(f"{indent}   └── No (>= {node.split_threshold:.1f}):")
            self.print_tree_structure(node.right_child, indent + "       ")

    def _parse_conditions(self, conditions: List[str]) -> Dict[str, str]:
        """解析条件列表，提取每个特征的条件"""
        result = {'BMI': '-', 'Weight': '-', 'Age': '-', 'Height': '-'}

        for cond in conditions:
            for feat in FEATURE_PRIORITY:
                if feat in cond:
                    # 提取条件部分
                    cond_part = cond.replace(feat, '').strip()
                    if result[feat] == '-':
                        result[feat] = cond_part
                    else:
                        result[feat] += ' & ' + cond_part
                    break

        return result

    def get_leaf_results(self) -> List[LeafNodeResult]:
        results = []
        for i, node in enumerate(self.leaf_nodes):
            cond_dict = self._parse_conditions(node.path_conditions)

            if node.optimal_week <= 12:
                rec = "Standard early screening (11-12w)"
            elif node.optimal_week <= 14:
                rec = "Slight delay (12-14w)"
            elif node.optimal_week <= 16:
                rec = "Delayed screening (14-16w)"
            else:
                rec = f"Sequential:  12w + retest at {node.optimal_week:.0f}w"

            results.append(LeafNodeResult(
                leaf_id=i,
                path_description=" AND ".join(node.path_conditions) if node.path_conditions else "Root",
                conditions=node.path_conditions,
                bmi_condition=cond_dict['BMI'],
                weight_condition=cond_dict['Weight'],
                age_condition=cond_dict['Age'],
                height_condition=cond_dict['Height'],
                sample_size=node.sample_size,
                pass_rate=node.pass_rate,
                optimal_week=node.optimal_week,
                compliance_probability=node.compliance_probability,
                risk_score=node.risk_score,
                ci_lower=node.ci_lower,
                ci_upper=node.ci_upper,
                recommendation=rec
            ))
        return results

    def generate_decision_table(self) -> pd.DataFrame:
        results = self.get_leaf_results()

        table_data = []
        for r in results:
            table_data.append({
                'Leaf_ID': r.leaf_id,
                'BMI': r.bmi_condition,
                'Weight': r.weight_condition,
                'Age': r.age_condition,
                'Height': r.height_condition,
                'Sample_Size': r.sample_size,
                'Pass_Rate(%)': round(r.pass_rate * 100, 1),
                'Optimal_Week': round(r.optimal_week, 1),
                'Compliance(%)': round(r.compliance_probability * 100, 1),
                'Risk_Score': round(r.risk_score, 3),
                '95%_CI': f"[{r.ci_lower:.1f}, {r.ci_upper:.1f}]",
                'Recommendation': r.recommendation
            })

        return pd.DataFrame(table_data)

    def print_decision_table(self):
        print("\n" + "=" * 70)
        print("📊 Complete Decision Table (Hierarchical:  BMI → Weight → Age → Height)")
        print("=" * 70)

        df = self.generate_decision_table()

        print("\n" + "-" * 150)
        print(f"{'ID':<4} {'BMI':<18} {'Weight':<15} {'Age':<15} {'Height':<15} "
              f"{'n':<6} {'Pass%':<8} {'Week':<8} {'Comply%':<10} {'Risk':<8}")
        print("-" * 150)

        for _, row in df.iterrows():
            bmi = row['BMI'][: 16] if len(row['BMI']) > 16 else row['BMI']
            weight = row['Weight'][: 13] if len(row['Weight']) > 13 else row['Weight']
            age = row['Age'][:13] if len(row['Age']) > 13 else row['Age']
            height = row['Height'][:13] if len(row['Height']) > 13 else row['Height']

            print(f"{row['Leaf_ID']:<4} {bmi:<18} {weight:<15} {age:<15} {height:<15} "
                  f"{row['Sample_Size']:<6} {row['Pass_Rate(%)']:<8.1f} {row['Optimal_Week']:<8.1f} "
                  f"{row['Compliance(%)']:<10.1f} {row['Risk_Score']:<8.3f}")
        print("-" * 150)

        return df


# ==========================================
# 可视化
# ==========================================
class TreeVisualizer:
    def __init__(self, tree: HierarchicalDecisionTree):
        self.tree = tree

    def _get_color(self, week: float) -> str:
        if week <= 12:
            return '#d4edda'
        elif week <= 14:
            return '#fff3cd'
        elif week <= 16:
            return '#ffeeba'
        return '#f8d7da'

    def plot_tree_diagram(self):
        fig, ax = plt.subplots(figsize=(22, 16))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis('off')

        self._draw_node(ax, self.tree.root, x=50, y=95, x_span=45)

        plt.title('Hierarchical Decision Tree for NIPT Optimal Timing\n'
                  '(Level 1: BMI → Level 2: Weight → Level 3: Age → Level 4: Height)',
                  fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('Q3_Decision_Tree.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("\n   ✓ Saved:  Q3_Decision_Tree.png")
        plt.close()

    def _draw_node(self, ax, node: TreeNode, x: float, y: float, x_span: float):
        if node.is_leaf:
            color = self._get_color(node.optimal_week)
            text = f"Leaf {node.node_id}\nn={node.sample_size}\nWeek={node.optimal_week:.1f}\nPass={node.pass_rate:.0%}"
            ax.text(x, y, text, ha='center', va='center', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, edgecolor='black'))
        else:
            # 根据特征选择颜色
            feat_colors = {'BMI': '#e74c3c', 'Weight': '#3498db', 'Age': '#2ecc71', 'Height': '#9b59b6'}
            color = feat_colors.get(node.split_feature, COLORS['info'])

            text = f"{node.split_feature}\n< {node.split_threshold:.1f}"
            ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor=color, edgecolor='black', alpha=0.9),
                    color='white')

            child_y = y - 18
            left_x = x - x_span / 2
            right_x = x + x_span / 2

            ax.annotate('', xy=(left_x, child_y + 6), xytext=(x, y - 4),
                        arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
            ax.annotate('', xy=(right_x, child_y + 6), xytext=(x, y - 4),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

            ax.text((x + left_x) / 2 - 2, (y + child_y) / 2, 'Y', fontsize=7, color='green', fontweight='bold')
            ax.text((x + right_x) / 2 + 2, (y + child_y) / 2, 'N', fontsize=7, color='red', fontweight='bold')

            if node.left_child:
                self._draw_node(ax, node.left_child, left_x, child_y, x_span / 2)
            if node.right_child:
                self._draw_node(ax, node.right_child, right_x, child_y, x_span / 2)

    def plot_feature_importance(self):
        feature_stats = {f: {'count': 0, 'score': 0} for f in FEATURE_PRIORITY}

        for split in self.tree.split_history:
            f = split['feature']
            feature_stats[f]['count'] += 1
            feature_stats[f]['score'] += split['score']

        features = FEATURE_PRIORITY
        counts = [feature_stats[f]['count'] for f in features]
        scores = [feature_stats[f]['score'] for f in features]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

        ax1 = axes[0]
        bars1 = ax1.barh(features, counts, color=colors, edgecolor='black')
        ax1.set_xlabel('Number of Splits')
        ax1.set_title('(A) Feature Usage in Tree', fontsize=12, fontweight='bold')
        for bar, c in zip(bars1, counts):
            ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2, str(c), va='center')
        ax1.grid(True, alpha=0.3, axis='x')

        ax2 = axes[1]
        bars2 = ax2.barh(features, scores, color=colors, edgecolor='black')
        ax2.set_xlabel('Total Split Score')
        ax2.set_title('(B) Feature Importance (Split Quality)', fontsize=12, fontweight='bold')
        for bar, s in zip(bars2, scores):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f'{s:.2f}', va='center')
        ax2.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig('Q3_Feature_Importance.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("   ✓ Saved:  Q3_Feature_Importance.png")
        plt.close()

    def plot_leaf_comparison(self):
        results = sorted(self.tree.get_leaf_results(), key=lambda x: x.optimal_week)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax1 = axes[0]
        ids = [f"L{r.leaf_id}" for r in results]
        weeks = [r.optimal_week for r in results]
        sizes = [r.sample_size for r in results]
        colors = [self._get_color(w) for w in weeks]

        bars = ax1.barh(ids, weeks, color=colors, edgecolor='black')
        for bar, n in zip(bars, sizes):
            ax1.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2, f'n={n}', va='center', fontsize=8)

        ax1.axvline(x=12, color='green', linestyle='--', lw=2, label='12w')
        ax1.axvline(x=16, color='orange', linestyle='--', lw=2, label='16w')
        ax1.set_xlabel('Optimal Week')
        ax1.set_title('(A) Optimal Week by Leaf Node', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='x')

        ax2 = axes[1]
        pass_rates = [r.pass_rate * 100 for r in results]
        scatter = ax2.scatter(weeks, pass_rates, s=[n * 2 for n in sizes], c=weeks, cmap='RdYlGn_r',alpha=0.7, edgecolors='black')
        for r in results:
            ax2.annotate(f'L{r.leaf_id}', (r.optimal_week, r.pass_rate * 100), textcoords="offset points", xytext=(5, 5), fontsize=7)
        ax2.set_xlabel('Optimal Week')
        ax2.set_ylabel('Pass Rate (%)')
        ax2.set_title('(B) Pass Rate vs Optimal Week', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Optimal Week')

        plt.tight_layout()
        plt.savefig('Q3_Leaf_Comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("   ✓ Saved: Q3_Leaf_Comparison.png")
        plt.close()

    def plot_threshold_distributions(self):
        """绘制各特征的阈值分布"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

        for i, feature in enumerate(FEATURE_PRIORITY):
            ax = axes[i // 2, i % 2]

            thresholds = [s['threshold'] for s in self.tree.split_history if s['feature'] == feature]
            feature_vals = self.tree.df[feature].dropna().values

            ax.hist(feature_vals, bins=30, alpha=0.6, color=colors[i],
                    edgecolor='black', label='Data Distribution')

            for j, t in enumerate(thresholds):
                ax.axvline(x=t, color='red', linestyle='--', lw=2,
                           label=f'Split:  {t:.1f}' if j == 0 else '')

            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.set_title(f'{feature} Distribution & Split Points', fontsize=11, fontweight='bold')

            if thresholds:
                ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('Q3_Threshold_Distributions.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("   ✓ Saved: Q3_Threshold_Distributions.png")
        plt.close()


# ==========================================
# Bootstrap分析
# ==========================================
class BootstrapAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def run_bootstrap(self, n_bootstrap: int = 100) -> pd.DataFrame:
        print("\n" + "=" * 70)
        print("📊 Bootstrap Sensitivity Analysis")
        print("=" * 70)

        y_vals = self.df['Y染色体浓度'].dropna()
        noise_sigma = 1.4826 * np.median(np.abs(y_vals - np.median(y_vals)))
        print(f"\n   Noise σ = {noise_sigma:.4f}")
        print(f"   Running {n_bootstrap} iterations...")

        threshold_results = {f: [] for f in FEATURE_PRIORITY}

        for b in range(n_bootstrap):
            if (b + 1) % 4 == 0:
                print(f"   Progress: {b + 1}/{n_bootstrap}")

            df_boot = self.df.sample(n=len(self.df), replace=True).reset_index(drop=True)
            noise = np.random.normal(0, noise_sigma, len(df_boot))
            df_boot['Y染色体浓度'] = (df_boot['Y染色体浓度'] + noise).clip(lower=0)

            try:
                tree_boot = HierarchicalDecisionTree(df_boot)
                tree_boot.node_counter = 0
                tree_boot.split_history = []
                tree_boot.leaf_nodes = []

                all_idx = np.arange(len(tree_boot.df))
                tree_boot.root = tree_boot._create_node(all_idx, 0, [])
                tree_boot._build_tree_recursive(tree_boot.root)

                for split in tree_boot.split_history:
                    threshold_results[split['feature']].append(split['threshold'])
            except:
                continue

        print("\n【Threshold Stability Analysis】")
        print("-" * 90)
        print(f"{'Feature':<10} {'Mean':<12} {'Std':<12} {'Median':<12} {'95% CI':<25} {'N_Splits':<10}")
        print("-" * 90)

        summary = []
        for f in FEATURE_PRIORITY:
            vals = threshold_results[f]
            if len(vals) >= 10:
                summary.append({
                    'Feature': f,
                    'Mean': np.mean(vals),
                    'Std': np.std(vals),
                    'Median': np.median(vals),
                    'CI_Lower': np.percentile(vals, 2.5),
                    'CI_Upper': np.percentile(vals, 97.5),
                    'N_Splits': len(vals)
                })
                print(f"{f:<10} {np.mean(vals):<12.2f} {np.std(vals):<12.2f} "
                      f"{np.median(vals):<12.2f} [{np.percentile(vals, 2.5):.1f}, {np.percentile(vals, 97.5):.1f}]{'':<5} {len(vals):<10}")
            else:
                print(f"{f:<10} {'Insufficient data':<50} {len(vals):<10}")

        print("-" * 90)
        return pd.DataFrame(summary)


# ==========================================
# 临床建议生成
# ==========================================
class ClinicalRecommendations:
    def __init__(self, tree: HierarchicalDecisionTree):
        self.tree = tree

    def generate(self):
        print("\n" + "=" * 70)
        print("📋 Clinical Recommendations")
        print("=" * 70)

        results = self.tree.get_leaf_results()

        # 分类
        categories = {
            'early': [r for r in results if r.optimal_week <= 12],
            'slight_delay': [r for r in results if 12 < r.optimal_week <= 14],
            'delay': [r for r in results if 14 < r.optimal_week <= 16],
            'sequential': [r for r in results if r.optimal_week > 16]
        }

        cat_names = {
            'early': '1.  Standard Early Screening (≤12 weeks)',
            'slight_delay': '2. Slight Delay (12-14 weeks)',
            'delay': '3. Delayed Screening (14-16 weeks)',
            'sequential': '4. Sequential Testing Required (>16 weeks)'
        }

        for key, name in cat_names.items():
            print(f"\n【{name}】")
            print("-" * 70)
            cat_results = categories[key]
            if cat_results:
                total_n = sum(r.sample_size for r in cat_results)
                print(f"   Groups: {len(cat_results)}, Total Samples: {total_n}")
                print(f"   Typical Patient Profiles:")
                for r in cat_results[: 4]:
                    profile_parts = []
                    if r.bmi_condition != '-':
                        profile_parts.append(f"BMI {r.bmi_condition}")
                    if r.weight_condition != '-':
                        profile_parts.append(f"Weight {r.weight_condition}")
                    if r.age_condition != '-':
                        profile_parts.append(f"Age {r.age_condition}")
                    if r.height_condition != '-':
                        profile_parts.append(f"Height {r.height_condition}")

                    profile = " & ".join(profile_parts) if profile_parts else "All patients"
                    print(f"   • {profile}")
                    print(f"     → Week={r.optimal_week:.1f}, n={r.sample_size}, Pass={r.pass_rate:.1%}")
            else:
                print("   No patient groups in this category")

        # 发现的最优阈值
        print("\n【Discovered Optimal Thresholds】")
        print("-" * 70)
        for feat in FEATURE_PRIORITY:
            thresholds = [s['threshold'] for s in self.tree.split_history if s['feature'] == feat]
            if thresholds:
                unique_thresh = sorted(set([round(t, 1) for t in thresholds]))
                print(f"   {feat}: {unique_thresh}")

        # 关键发现
        print("\n【Key Findings】")
        print("-" * 70)

        # BMI阈值
        bmi_thresholds = [s['threshold'] for s in self.tree.split_history if s['feature'] == 'BMI']
        if bmi_thresholds:
            print(f"   • BMI is the primary stratification factor")
            print(f"     Critical threshold(s): {sorted(set([round(t, 1) for t in bmi_thresholds]))}")

        # 高风险组
        high_risk = [r for r in results if r.optimal_week > 14]
        if high_risk:
            print(f"   • {len(high_risk)} patient groups require delayed/sequential testing")
            worst = max(high_risk, key=lambda x: x.optimal_week)
            print(f"     Highest risk group:  Optimal week = {worst.optimal_week:.1f}")

        # 特征重要性
        print("\n【Feature Importance Ranking】")
        print("-" * 70)
        feat_scores = {}
        for s in self.tree.split_history:
            f = s['feature']
            feat_scores[f] = feat_scores.get(f, 0) + s['score']

        for i, (f, score) in enumerate(sorted(feat_scores.items(), key=lambda x: -x[1]), 1):
            print(f"   {i}. {f}: {score:.3f}")


# ==========================================
# 主程序
# ==========================================
def main():
    print("\n" + "🎯" * 35)
    print("NIPT Optimal Timing - Hierarchical Decision Tree")
    print("Feature Priority:  BMI → Weight → Age → Height")
    print("🎯" * 35)
    print(f"\nMax Depth: {TREE_PARAMS['max_depth']}")
    print(f"Min Leaf Samples: {TREE_PARAMS['min_samples_leaf']}")

    # 加载数据
    processor = NIPTDataProcessor('附件.xlsx')
    df_male, df_female = processor.load_and_process()

    if df_male is None:
        return

    # 构建树
    print("\n" + "📊" * 35)
    print("Building Hierarchical Decision Tree")
    print("📊" * 35)

    tree = HierarchicalDecisionTree(df_male)
    tree.fit()

    # 输出结果
    tree.print_split_history()
    tree.print_tree_structure()
    table = tree.print_decision_table()

    table.to_excel('Q3_Decision_Table.xlsx', index=False)
    print("\n   ✓ Saved: Q3_Decision_Table.xlsx")

    # 可视化
    print("\n" + "=" * 70)
    print("📊 Generating Visualizations")
    print("=" * 70)

    viz = TreeVisualizer(tree)
    viz.plot_tree_diagram()
    viz.plot_feature_importance()
    viz.plot_leaf_comparison()
    viz.plot_threshold_distributions()

    # Bootstrap分析
    bootstrap = BootstrapAnalyzer(df_male)
    bootstrap.run_bootstrap(n_bootstrap=100)

    # 临床建议
    recs = ClinicalRecommendations(tree)
    recs.generate()

    print("\n" + "=" * 70)
    print("✅ Analysis Complete!")
    print("=" * 70)
    print("\nGenerated Files:")
    print("   1. Q3_Decision_Table. xlsx - Complete decision table")
    print("   2. Q3_Decision_Tree.png - Tree visualization")
    print("   3. Q3_Feature_Importance.png - Feature importance")
    print("   4. Q3_Leaf_Comparison.png - Leaf node comparison")
    print("   5. Q3_Threshold_Distributions.png - Threshold distributions")


if __name__ == "__main__":
    main()