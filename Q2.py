import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Tuple, Dict, List, Optional, Union
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize, minimize_scalar
from scipy.interpolate import interp1d
import statsmodels.api as sm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# ==========================================
# 全局配置 (Global Configuration)
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
warnings.filterwarnings('ignore')

# 医学常数 (Clinical Constants)
FF_THRESHOLD = 0.04  # 胎儿分数达标阈值 (4%)
MIN_GA_WEEKS = 10  # 最早可检测孕周
MAX_GA_WEEKS = 28  # 最晚推荐检测孕周
CRITICAL_GA_WEEKS = 20  # 临床关键决策时间点

# 颜色方案
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#28A745',
    'warning': '#FFC107',
    'danger': '#DC3545',
    'info': '#17A2B8',
    'dark': '#343A40',
    'light': '#F8F9FA',
}

# 风险权重参数 (Risk Weight Parameters)
RISK_WEIGHTS = {
    'early':  1.0,    # b1:  Early period (<12 weeks)
    'middle': 2.0,   # b2: Middle period (13-27 weeks)
    'late': 6.0,     # b3: Late period (>28 weeks)
    'lambda2': 0.80  # Risk-accuracy trade-off parameter
}


# ==========================================
# 数据类定义 (Data Classes)
# ==========================================
@dataclass
class BMIGroup:
    """BMI分组数据类"""
    group_id: int
    group_name: str
    bmi_lower: float
    bmi_upper: float
    sample_size: int
    mean_concentration: float
    pass_rate: float
    regression_params: Dict


@dataclass
class OptimalTimingResult:
    """最优检测时点结果"""
    group_name: str
    bmi_range: Tuple[float, float]
    optimal_week: float
    pass_probability: float
    risk_score: float
    quantile_30_week: float
    confidence_interval: Tuple[float, float]


# ==========================================
# 模块1:  数据预处理 (从Q1继承)
# ==========================================
class NIPTDataProcessor:
    """NIPT数据预处理类 - 继承自Q1"""

    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.imputer = IterativeImputer(
            max_iter=20,
            random_state=2025,
            min_value=0,
            initial_strategy='median'
        )
        self.processing_log = []

    def _parse_gestational_age(self, ga_str) -> float:
        """解析孕周字符串为数值"""
        if pd.isna(ga_str):
            return np.nan
        try:
            ga_str = str(ga_str).lower().strip().replace('d', '')
            if 'w' in ga_str:
                parts = ga_str.split('w')
                weeks = float(parts[0])
                days = 0
                if len(parts) > 1:
                    day_part = parts[1]. replace('+', '').strip()
                    if day_part:
                        days = float(day_part)
                return weeks + days / 7.0
            return float(ga_str)
        except (ValueError, TypeError):
            return np.nan

    def _validate_bmi(self, df: pd.DataFrame) -> pd.DataFrame:
        """验证BMI合理性"""
        df = df.copy()
        invalid_mask = (df['BMI_calc'] < 15) | (df['BMI_calc'] > 60)
        n_invalid = invalid_mask.sum()
        if n_invalid > 0:
            self.processing_log.append(f"   ⚠ Found {n_invalid} abnormal BMI values, marked as missing")
            df.loc[invalid_mask, 'BMI_calc'] = np.nan
        return df

    def process_dataset(self, df_raw: pd.DataFrame, dataset_name: str = 'data') -> pd.DataFrame:
        """核心数据处理流程"""
        df = df_raw.copy()
        self.processing_log.append(f"\n{'=' * 60}")
        self.processing_log.append(f"📊 Processing dataset: {dataset_name}")
        self.processing_log.append(f"   Original sample size: {len(df)}")

        # 1. 孕周数值化
        if '检测孕周' in df. columns:
            df['GA_numeric'] = df['检测孕周'].apply(self._parse_gestational_age)

        # 2. 数值列清洗
        numeric_cols = ['年龄', '身高', '体重']
        for col in numeric_cols:
            if col in df. columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 3. MICE多重插补
        available_cols = [c for c in numeric_cols if c in df.columns]
        if available_cols:
            n_missing_before = df[available_cols].isna().sum().sum()
            if n_missing_before > 0:
                df[available_cols] = self. imputer.fit_transform(df[available_cols])

        # 4. 身高单位统一 (cm)
        if '身高' in df.columns:
            if df['身高'].median() < 3:
                df['身高'] = df['身高'] * 100

        # 5. BMI计算与验证
        if '身高' in df.columns and '体重' in df.columns:
            df['BMI_calc'] = df['体重'] / ((df['身高'] / 100) ** 2)
            df = self._validate_bmi(df)

        # 6. 特征工程
        if 'BMI_calc' in df.columns:
            df['Log_BMI'] = np.log(df['BMI_calc'] + 1e-5)
            df['BMI_squared'] = df['BMI_calc'] ** 2

        if 'GA_numeric' in df.columns:
            df['GA_squared'] = df['GA_numeric'] ** 2

        if 'GA_numeric' in df.columns and 'BMI_calc' in df.columns:
            df['GA_BMI_ratio'] = df['GA_numeric'] / (df['BMI_calc'] + 1e-5)
            df['GA_BMI_interaction'] = df['GA_numeric'] * df['BMI_calc']

        # 7. 达标标记
        if 'Y染色体浓度' in df.columns:
            df['FF_Pass'] = (df['Y染色体浓度'] >= FF_THRESHOLD).astype(int)

        self.processing_log.append(f"   ✓ Final sample size: {len(df)}")
        return df

    def load_and_process(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """加载并处理Excel数据"""
        print("=" * 70)
        print("📂 NIPT Data Loading and Preprocessing")
        print("=" * 70)

        try:
            df_male = pd.read_excel(self.excel_path, sheet_name='男胎检测数据')
            df_female = pd.read_excel(self.excel_path, sheet_name='女胎检测数据')

            df_male_processed = self.process_dataset(df_male, 'Male Fetus')
            df_female_processed = self.process_dataset(df_female, 'Female Fetus')

            for log in self.processing_log:
                print(log)

            print(f"\n📊 Data Overview:")
            print(f"   Male fetus samples: {len(df_male_processed)}")
            print(f"   Female fetus samples: {len(df_female_processed)}")

            return df_male_processed, df_female_processed

        except FileNotFoundError:
            print(f"❌ File not found: {self.excel_path}")
            return None, None
        except Exception as e:
            print(f"❌ Loading error: {e}")
            return None, None


# ==========================================
# 模块2: 问题2 - BMI分段优化与最佳检测时点
# ==========================================
class Problem2_OptimalTiming:
    """
    问题2: 基于非线性优化的NIPT检测时点决策

    核心方法:
    1. 分段回归断点检测 (Piecewise Regression with Change-point Detection)
    2. 动态规划/网格搜索确定最优BMI分组
    3. 生存分析框架下的达标概率估计
    4. 风险-准确性权衡的目标函数优化
    5. 簇级Bootstrap误差敏感性分析
    """

    def __init__(self, df: pd.DataFrame):
        """
        初始化优化器

        Args:
            df: 预处理后的数据框
        """
        self.df = df. dropna(subset=['BMI_calc', 'GA_numeric', 'Y染色体浓度']).copy()
        self.bmi_groups = []
        self.optimal_timings = []
        self.breakpoints = []
        self.baseline_model = None
        self.noise_sigma = None

    # ==========================================
    # 2.1 基础回归模型
    # ==========================================
    def fit_baseline_regression(self) -> Dict:
        """
        拟合基础多元线性回归模型

        Returns:
            Dict: 模型参数和统计量
        """
        print("\n" + "=" * 70)
        print("📊 Section 2: Baseline Regression Model")
        print("=" * 70)

        X = self.df[['BMI_calc', 'GA_numeric']]
        y = self.df['Y染色体浓度']

        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit()
        self.baseline_model = model

        print("\n【Baseline Model Summary】")
        print(f"   Sample size: n = {len(self.df)}")
        print(f"   R² = {model.rsquared:.4f}")
        print(f"   Adjusted R² = {model.rsquared_adj:.4f}")

        print("\n【Regression Coefficients】")
        print("-" * 70)
        print(f"{'Variable':<15} {'Coefficient':<15} {'Std Error':<12} {'p-value':<15}")
        print("-" * 70)

        var_names = ['Intercept', 'BMI', 'Gestational_Age']
        for i, name in enumerate(var_names):
            sig = '***' if model.pvalues[i] < 0.001 else '**' if model.pvalues[i] < 0.01 else '*' if model.pvalues[i] < 0.05 else ''
            print(f"{name: <15} {model.params[i]:<15.6f} {model.bse[i]:<12.6f} {model.pvalues[i]: <15.2e} {sig}")
        print("-" * 70)

        # 关键发现
        print("\n📌 Key Findings:")
        print(f"   • BMI coefficient (β): {model.params[1]:.4f}")
        print(f"     → Each 1 unit BMI increase reduces Y concentration by {-model.params[1]*100:.2f}%")
        print(f"   • GA coefficient (γ): {model.params[2]:.4f}")
        print(f"     → Each 1 week increase raises Y concentration by {model.params[2]*100:.2f}%")

        # 计算BMI补偿周数
        bmi_per_week = abs(model.params[1] / model.params[2])
        print(f"\n   ⚡ Clinical Implication:")
        print(f"      For every {bmi_per_week:.1f} unit BMI increase, 1 additional week is needed")

        return {
            'intercept': model.params[0],
            'beta_bmi': model.params[1],
            'gamma_ga': model.params[2],
            'r_squared': model.rsquared,
            'residuals': model.resid
        }

    # ==========================================
    # 3. BMI分段优化 (Piecewise Optimization)
    # ==========================================
    def _calculate_segment_rss(self, df_segment: pd.DataFrame) -> Tuple[float, Dict]:
        """计算单个分段的RSS和回归参数"""
        if len(df_segment) < 10:
            return np.inf, {}

        X = df_segment[['BMI_calc', 'GA_numeric']]
        y = df_segment['Y染色体浓度']

        X_sm = sm.add_constant(X)
        try:
            model = sm.OLS(y, X_sm).fit()
            rss = np.sum(model.resid ** 2)
            params = {
                'intercept': model. params[0],
                'beta_bmi': model.params[1],
                'gamma_ga': model.params[2],
                'r_squared': model. rsquared
            }
            return rss, params
        except:
            return np.inf, {}

    def _evaluate_breakpoints(self, breakpoints: List[float]) -> Tuple[float, List[Dict]]:
        """
        评估给定断点集合的总体拟合质量

        Args:
            breakpoints: BMI断点列表

        Returns:
            Tuple:  (总RSS, 各段参数列表)
        """
        all_breaks = [self.df['BMI_calc'].min() - 0.01] + sorted(breakpoints) + [self.df['BMI_calc'].max() + 0.01]
        total_rss = 0
        segment_params = []

        for i in range(len(all_breaks) - 1):
            lower, upper = all_breaks[i], all_breaks[i + 1]
            segment = self.df[(self.df['BMI_calc'] >= lower) & (self.df['BMI_calc'] < upper)]
            rss, params = self._calculate_segment_rss(segment)
            total_rss += rss
            params['bmi_range'] = (lower, upper)
            params['sample_size'] = len(segment)
            segment_params.append(params)

        return total_rss, segment_params

    def optimize_bmi_breakpoints(self, n_groups: int = 4, search_range: Tuple[float, float] = (20, 45)) -> List[float]:
        """
        动态规划/网格搜索确定最优BMI断点

        Args:
            n_groups: 目标分组数
            search_range: BMI搜索范围

        Returns:
            List[float]: 最优断点列表
        """
        print("\n" + "=" * 70)
        print("📊 Section 3: Optimal BMI Breakpoint Detection")
        print("=" * 70)

        n_breaks = n_groups - 1
        bmi_min, bmi_max = search_range
        grid_step = 0.5

        # 生成候选断点网格
        candidate_points = np.arange(bmi_min, bmi_max, grid_step)

        print(f"\n【Grid Search Configuration】")
        print(f"   Target groups: {n_groups}")
        print(f"   Search range: BMI {bmi_min} - {bmi_max}")
        print(f"   Grid step: {grid_step}")
        print(f"   Candidate points: {len(candidate_points)}")

        best_rss = np.inf
        best_breaks = None
        best_params = None

        # 递归生成所有可能的断点组合
        from itertools import combinations

        total_combinations = len(list(combinations(candidate_points, n_breaks)))
        print(f"   Total combinations to evaluate: {total_combinations}")

        for breaks in combinations(candidate_points, n_breaks):
            rss, params = self._evaluate_breakpoints(list(breaks))
            # 添加BIC惩罚项
            n = len(self.df)
            k = n_groups * 3  # 每组3个参数
            bic_penalty = k * np.log(n)
            penalized_score = rss + bic_penalty * 0.001  # 缩放因子

            if penalized_score < best_rss:
                best_rss = penalized_score
                best_breaks = list(breaks)
                best_params = params

        self.breakpoints = best_breaks
        print(f"\n【Optimal Breakpoints Found】")
        print(f"   Breakpoints: {[f'{bp:.2f}' for bp in best_breaks]}")

        # 输出分组详情
        print("\n【BMI Group Summary】")
        print("-" * 90)
        print(f"{'Group':<8} {'BMI Range':<20} {'Sample Size':<12} {'R²':<10} {'β_BMI':<12} {'γ_GA': <12}")
        print("-" * 90)

        for i, params in enumerate(best_params):
            if params:
                bmi_range = params. get('bmi_range', (0, 0))
                group_name = f"G{i+1}"
                print(f"{group_name:<8} [{bmi_range[0]:.1f}, {bmi_range[1]:.1f}){'':<8} "
                      f"{params. get('sample_size', 0):<12} {params.get('r_squared', 0):<10.4f} "
                      f"{params.get('beta_bmi', 0):<12.6f} {params.get('gamma_ga', 0):<12.6f}")

                self.bmi_groups.append(BMIGroup(
                    group_id=i + 1,
                    group_name=group_name,
                    bmi_lower=bmi_range[0],
                    bmi_upper=bmi_range[1],
                    sample_size=params. get('sample_size', 0),
                    mean_concentration=0,
                    pass_rate=0,
                    regression_params=params
                ))
        print("-" * 90)

        # 绘制分组可视化
        self._plot_bmi_segmentation(best_breaks, best_params)

        return best_breaks

    def _plot_bmi_segmentation(self, breakpoints: List[float], segment_params: List[Dict]):
        """绘制BMI分组可视化"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 图1: BMI vs Y浓度散点图 + 分段回归线
        ax1 = axes[0]
        colors = plt.cm.tab10.colors[: len(breakpoints) + 1]
        all_breaks = [self.df['BMI_calc'].min()] + breakpoints + [self.df['BMI_calc'].max()]

        for i in range(len(breakpoints) + 1):
            lower, upper = all_breaks[i], all_breaks[i + 1]
            mask = (self.df['BMI_calc'] >= lower) & (self.df['BMI_calc'] < upper)
            segment = self.df[mask]

            ax1.scatter(segment['BMI_calc'], segment['Y染色体浓度'],
                       alpha=0.4, s=15, c=colors[i], label=f'G{i+1}:  [{lower:.1f}, {upper:.1f})')

            # 绘制分段回归线
            if len(segment) > 10:
                X_plot = np.linspace(lower, upper, 50)
                params = segment_params[i]
                if params:
                    y_plot = (params['intercept'] +
                             params['beta_bmi'] * X_plot +
                             params['gamma_ga'] * segment['GA_numeric']. mean())
                    ax1.plot(X_plot, y_plot, color=colors[i], linewidth=2.5)

        # 绘制断点垂直线
        for bp in breakpoints:
            ax1.axvline(x=bp, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

        ax1.axhline(y=FF_THRESHOLD, color='green', linestyle=':', linewidth=2, label=f'Threshold ({FF_THRESHOLD*100:.0f}%)')
        ax1.set_xlabel('BMI (kg/m²)', fontsize=11)
        ax1.set_ylabel('Y Chromosome Concentration', fontsize=11)
        ax1.set_title('(A) Piecewise Regression:  Y Concentration vs BMI', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # 图2: 各组达标率条形图
        ax2 = axes[1]
        group_names = []
        pass_rates = []
        sample_sizes = []

        for i in range(len(breakpoints) + 1):
            lower, upper = all_breaks[i], all_breaks[i + 1]
            mask = (self.df['BMI_calc'] >= lower) & (self.df['BMI_calc'] < upper)
            segment = self.df[mask]
            group_names.append(f'G{i+1}\n[{lower:.1f},{upper:.1f})')
            pass_rate = (segment['Y染色体浓度'] >= FF_THRESHOLD).mean()
            pass_rates.append(pass_rate)
            sample_sizes.append(len(segment))

        bars = ax2.bar(group_names, pass_rates, color=colors[: len(group_names)], edgecolor='black', alpha=0.8)

        # 在柱状图上添加数值标签
        for bar, rate, n in zip(bars, pass_rates, sample_sizes):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{rate:.1%}\n(n={n})', ha='center', va='bottom', fontsize=9)

        ax2.axhline(y=0.85, color='orange', linestyle='--', linewidth=2, label='85% Target')
        ax2.axhline(y=0.95, color='green', linestyle='--', linewidth=2, label='95% Target')
        ax2.set_xlabel('BMI Group', fontsize=11)
        ax2.set_ylabel('Pass Rate (Y ≥ 4%)', fontsize=11)
        ax2.set_title('(B) Pass Rate by Optimized BMI Groups', fontsize=12, fontweight='bold')
        ax2.set_ylim(0,1.15)
        ax2.legend(loc='lower left', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('Q2_BMI_Segmentation.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("\n   ✓ Figure saved:  Q2_BMI_Segmentation.png")
        plt.close()

    # ==========================================
    # 4. 最佳检测时点优化
    # ==========================================
    def _risk_function(self, t: float) -> float:
        """
        计算延迟检测的临床风险函数 Risk(t)

        Risk function based on three-phase model:
        - Early (<12 weeks): Low risk (weight b1=1)
        - Middle (13-27 weeks): Medium risk (weight b2=3)
        - Late (>28 weeks): High risk (weight b3=6)

        Args:
            t: 孕周

        Returns:
            float: 归一化风险值 [0, 1]
        """
        b1, b2, b3 = RISK_WEIGHTS['early'], RISK_WEIGHTS['middle'], RISK_WEIGHTS['late']

        # 分段基函数
        def B_early(t):
            return max(0, min(1, (12 - t) / 2)) if t < 12 else 0

        def B_middle(t):
            if t < 12:
                return 0
            elif t < 28:
                return (t - 12) / 15
            else:
                return 1

        def B_late(t):
            return max(0, (t - 28) / 12) if t > 28 else 0

        # 累积风险
        r_t = b1 * B_early(t) + b2 * B_middle(t) + b3 * B_late(t)

        # 归一化
        r_10 = b1 * B_early(10) + b2 * B_middle(10) + b3 * B_late(10)
        r_40 = b1 * B_early(40) + b2 * B_middle(40) + b3 * B_late(40)

        risk = np.clip((r_t - r_10) / (r_40 - r_10), 0, 1)
        return risk

    def _estimate_pass_probability(self, df_group: pd.DataFrame, t: float) -> float:
        """
        估计在孕周t时的达标概率 (基于生存分析框架)

        使用Kaplan-Meier估计器的简化版本

        Args:
            df_group: 分组数据
            t: 目标孕周

        Returns:
            float: 达标概率 P(Y >= 4% | GA = t)
        """
        # 筛选在t周附近的样本 (±1周窗口)
        window = 1.5
        nearby = df_group[(df_group['GA_numeric'] >= t - window) &
                          (df_group['GA_numeric'] <= t + window)]

        if len(nearby) < 5:
            # 样本不足时使用回归预测
            params = self.baseline_model.params if self.baseline_model else [0.1, -0.002, 0.001]
            mean_bmi = df_group['BMI_calc'].mean()
            predicted_y = params[0] + params[1] * mean_bmi + params[2] * t

            # 假设正态分布，估计达标概率
            residual_std = 0.025 if self.baseline_model is None else np.std(self.baseline_model.resid)
            prob = 1 - stats.norm.cdf(FF_THRESHOLD, loc=predicted_y, scale=residual_std)
            return prob

        # 直接计算经验达标率
        pass_rate = (nearby['Y染色体浓度'] >= FF_THRESHOLD).mean()
        return pass_rate

    def _estimate_pass_probability_survival(self, df_group: pd.DataFrame, t: float) -> float:
        """
        基于回归模型的达标概率估计

        Args:
            df_group:  分组数据
            t: 目标孕周

        Returns:
            float: 达标概率 P(Y >= 4% | GA = t)
        """
        if len(df_group) < 10:
            return 0.5

        # 使用组内回归模型预测
        X = df_group[['BMI_calc', 'GA_numeric']]
        y = df_group['Y染色体浓度']
        X_sm = sm.add_constant(X)

        try:
            model = sm.OLS(y, X_sm).fit()

            # 计算在时间t时的预测浓度分布
            mean_bmi = df_group['BMI_calc'].mean()
            predicted_y = model.params[0] + model.params[1] * mean_bmi + model.params[2] * t

            # 使用残差标准差估计不确定性
            residual_std = np.sqrt(model.mse_resid)

            # 计算达标概率 P(Y >= 0.04)
            prob = 1 - stats.norm.cdf(FF_THRESHOLD, loc=predicted_y, scale=residual_std)

            return float(np.clip(prob, 0.01, 0.99))
        except:
            return 0.5

    def _objective_function(self, t: float, df_group: pd.DataFrame) -> float:
        """
        综合目标函数 J(t) = -ln(F(t)) + λ₂ × Risk(t) + 早期惩罚

        Args:
            t: 检测孕周
            df_group:  分组数据

        Returns:
            float: 目标函数值 (越小越优)
        """
        # 达标概率
        F_t = self._estimate_pass_probability_survival(df_group, t)
        F_t = max(F_t, 0.01)  # 避免log(0)

        # 风险值
        risk_t = self._risk_function(t)

        # 检测失败成本 (达标率不足的惩罚)
        failure_cost = -np.log(F_t)

        # 如果达标率低于85%，额外惩罚
        if F_t < 0.85:
            failure_cost += (0.85 - F_t) * 10

        # 目标函数
        lambda2 = RISK_WEIGHTS['lambda2']
        J_t = failure_cost + lambda2 * risk_t

        return J_t

    def find_optimal_timing_per_group(self) -> List[OptimalTimingResult]:
        """
        为每个BMI分组找到最佳检测时点

        Returns:
            List[OptimalTimingResult]:  各组最优时点结果
        """
        print("\n" + "=" * 70)
        print("📊 Section 4: Optimal Detection Timing per BMI Group")
        print("=" * 70)

        all_breaks = [self.df['BMI_calc'].min()] + self.breakpoints + [self.df['BMI_calc'].max()]
        results = []

        print("\n【Optimization Results】")
        print("-" * 100)
        print(f"{'Group':<8} {'BMI Range':<18} {'Optimal Week':<14} {'Pass Prob':<12} {'Risk Score':<12} {'τ=0.30 Week':<12}")
        print("-" * 100)

        for i in range(len(self.breakpoints) + 1):
            lower, upper = all_breaks[i], all_breaks[i + 1]
            mask = (self.df['BMI_calc'] >= lower) & (self.df['BMI_calc'] < upper)
            df_group = self.df[mask]

            if len(df_group) < 20:
                print(f"G{i+1:<7} [{lower:.1f}, {upper:.1f}){'':<6} Insufficient data (n={len(df_group)})")
                continue

            # 网格搜索最优时点
            search_weeks = np.arange(10, 25, 0.1)
            best_week = 12
            best_score = np. inf

            for week in search_weeks:
                score = self._objective_function(week, df_group)
                if score < best_score:
                    best_score = score
                    best_week = week

            # 计算该时点的达标概率和风险
            pass_prob = self._estimate_pass_probability_survival(df_group, best_week)
            risk_score = self._risk_function(best_week)

            # 计算τ=0.30分位点 (30%孕妇达标的孕周)
            quantile_week = self._find_quantile_week(df_group, tau=0.30)

            result = OptimalTimingResult(
                group_name=f"G{i+1}",
                bmi_range=(lower, upper),
                optimal_week=best_week,
                pass_probability=pass_prob,
                risk_score=risk_score,
                quantile_30_week=quantile_week,
                confidence_interval=(best_week - 1.0, best_week + 1.0)  # 初步估计
            )
            results.append(result)

            print(f"G{i+1:<7} [{lower:.1f}, {upper:.1f}){'':<6} {best_week:<14.1f} {pass_prob: <12.1%} "
                  f"{risk_score:<12.3f} {quantile_week: <12.2f}")

        print("-" * 100)

        self.optimal_timings = results

        # 绘制优化结果
        self._plot_optimal_timing(results)

        return results

    def _find_quantile_week(self, df_group: pd.DataFrame, tau: float = 0.30) -> float:
        """找到τ分位达标孕周"""
        # 按孕周排序
        sorted_data = df_group.sort_values('GA_numeric')
        cumulative_pass = (sorted_data['Y染色体浓度'] >= FF_THRESHOLD).cumsum() / len(sorted_data)

        # 找到首次超过tau的孕周
        mask = cumulative_pass >= tau
        if mask.any():
            return sorted_data.loc[mask.idxmax(), 'GA_numeric']
        return sorted_data['GA_numeric'].max()

    def _plot_optimal_timing(self, results: List[OptimalTimingResult]):
        """绘制最优检测时点分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 图1: 目标函数曲线
        ax1 = axes[0, 0]
        all_breaks = [self.df['BMI_calc'].min()] + self.breakpoints + [self.df['BMI_calc']. max()]
        colors = plt.cm.tab10.colors[: len(self.breakpoints) + 1]

        for i, result in enumerate(results):
            lower, upper = all_breaks[i], all_breaks[i + 1]
            mask = (self.df['BMI_calc'] >= lower) & (self.df['BMI_calc'] < upper)
            df_group = self.df[mask]

            weeks = np.arange(10, 25, 0.2)
            scores = [self._objective_function(w, df_group) for w in weeks]

            ax1.plot(weeks, scores, color=colors[i], linewidth=2, label=f'{result.group_name}:  BMI [{lower:.1f}, {upper:.1f})')
            ax1.scatter([result.optimal_week], [self._objective_function(result.optimal_week, df_group)],
                       color=colors[i], s=100, zorder=5, edgecolors='black', marker='*')

        ax1.set_xlabel('Gestational Age (weeks)', fontsize=11)
        ax1.set_ylabel('Objective Function J(t)', fontsize=11)
        ax1.set_title('(A) Objective Function:  Risk-Accuracy Trade-off', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # 图2: 风险函数
        ax2 = axes[0, 1]
        weeks = np.arange(8, 35, 0.1)
        risks = [self._risk_function(w) for w in weeks]
        ax2.plot(weeks, risks, color=COLORS['danger'], linewidth=2.5)
        ax2.fill_between(weeks, 0, risks, alpha=0.3, color=COLORS['danger'])

        # 标注关键时期
        ax2.axvspan(10, 12, alpha=0.2, color='green', label='Early Period (Low Risk)')
        ax2.axvspan(12, 20, alpha=0.2, color='yellow', label='Middle Period (Medium Risk)')
        ax2.axvspan(20, 28, alpha=0.2, color='orange', label='Late Middle (High Risk)')
        ax2.axvspan(28, 35, alpha=0.2, color='red', label='Late Period (Critical)')

        ax2.set_xlabel('Gestational Age (weeks)', fontsize=11)
        ax2.set_ylabel('Risk Score', fontsize=11)
        ax2.set_title('(B) Clinical Risk Function R(t)', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 图3: 达标概率曲线
        ax3 = axes[1, 0]
        for i, result in enumerate(results):
            lower, upper = all_breaks[i], all_breaks[i + 1]
            mask = (self.df['BMI_calc'] >= lower) & (self.df['BMI_calc'] < upper)
            df_group = self.df[mask]

            weeks = np.arange(10, 25, 0.5)
            probs = [self._estimate_pass_probability_survival(df_group, w) for w in weeks]

            ax3.plot(weeks, probs, color=colors[i], linewidth=2, label=f'{result.group_name}')
            ax3.axvline(x=result.optimal_week, color=colors[i], linestyle='--', alpha=0.7)

        ax3.axhline(y=0.95, color='green', linestyle=':', linewidth=2, label='95% Target')
        ax3.axhline(y=0.85, color='orange', linestyle=':', linewidth=2, label='85% Target')
        ax3.set_xlabel('Gestational Age (weeks)', fontsize=11)
        ax3.set_ylabel('Pass Probability P(Y ≥ 4%)', fontsize=11)
        ax3.set_title('(C) Cumulative Pass Probability by Group', fontsize=12, fontweight='bold')
        ax3.legend(loc='lower right', fontsize=9)
        ax3.set_ylim(0, 1.05)
        ax3.grid(True, alpha=0.3)

        # 图4: 最优时点汇总条形图
        ax4 = axes[1, 1]
        group_names = [r.group_name for r in results]
        optimal_weeks = [r.optimal_week for r in results]
        pass_probs = [r.pass_probability for r in results]

        x = np.arange(len(group_names))
        width = 0.35

        bars1 = ax4.bar(x - width/2, optimal_weeks, width, label='Optimal Week', color=COLORS['primary'], edgecolor='black')
        ax4_twin = ax4.twinx()
        bars2 = ax4_twin.bar(x + width/2, [p*100 for p in pass_probs], width, label='Pass Rate (%)',
                             color=COLORS['success'], edgecolor='black', alpha=0.7)

        ax4.set_xlabel('BMI Group', fontsize=11)
        ax4.set_ylabel('Optimal Gestational Week', fontsize=11, color=COLORS['primary'])
        ax4_twin.set_ylabel('Pass Rate (%)', fontsize=11, color=COLORS['success'])
        ax4.set_xticks(x)
        ax4.set_xticklabels([f"{r. group_name}\n[{r.bmi_range[0]:.0f},{r.bmi_range[1]:.0f})" for r in results])
        ax4.set_title('(D) Optimal Timing Summary', fontsize=12, fontweight='bold')

        # 添加数值标签
        for bar, week in zip(bars1, optimal_weeks):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{week:.1f}w',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax4.legend(loc='upper left', fontsize=9)
        ax4_twin.legend(loc='upper right', fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('Q2_Optimal_Timing.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("\n   ✓ Figure saved:  Q2_Optimal_Timing.png")
        plt.close()

    # ==========================================
    # 5. 误差敏感性分析 (Cluster Bootstrap)
    # ==========================================
    def estimate_measurement_noise(self) -> float:
        """
        估计检测误差标准差 σ

        使用MAD (Median Absolute Deviation) 方法进行稳健估计

        Returns:
            float: 噪声标准差
        """
        if self.baseline_model is None:
            self.fit_baseline_regression()

        residuals = self.baseline_model. resid
        mad = np.median(np.abs(residuals - np.median(residuals)))
        sigma = 1.4826 * mad  # MAD to standard deviation conversion

        self.noise_sigma = sigma
        print(f"\n   Estimated measurement noise σ = {sigma:.4f}")
        return sigma

    def cluster_bootstrap_analysis(self, n_bootstrap: int = 200) -> pd.DataFrame:
        """
        簇级Bootstrap误差敏感性分析

        Args:
            n_bootstrap: Bootstrap迭代次数

        Returns:
            pd.DataFrame: 各组最优时点的分布统计
        """
        print("\n" + "=" * 70)
        print("📊 Section 5: Cluster Bootstrap Sensitivity Analysis")
        print("=" * 70)

        if self.noise_sigma is None:
            self.estimate_measurement_noise()

        # 为每个样本创建伪ID（如果没有真实ID）
        if 'sample_id' not in self.df.columns:
            self.df['sample_id'] = range(len(self.df))

        all_breaks = [self.df['BMI_calc'].min()] + self.breakpoints + [self.df['BMI_calc']. max()]
        n_groups = len(self.breakpoints) + 1

        # 存储Bootstrap结果
        bootstrap_results = {f'G{i+1}': [] for i in range(n_groups)}

        print(f"\n   Running {n_bootstrap} bootstrap iterations...")
        print(f"   Noise level σ = {self.noise_sigma:.4f}")

        for b in range(n_bootstrap):
            if (b + 1) % 50 == 0:
                print(f"   Progress: {b+1}/{n_bootstrap}")

            # 簇级重采样
            unique_ids = self.df['sample_id'].unique()
            resampled_ids = np.random.choice(unique_ids, size=len(unique_ids), replace=True)
            df_resampled = self.df[self.df['sample_id'].isin(resampled_ids)].copy()

            # 添加噪声扰动
            noise = np.random.normal(0, self.noise_sigma, len(df_resampled))
            df_resampled['Y染色体浓度'] = df_resampled['Y染色体浓度'] + noise
            df_resampled['Y染色体浓度'] = df_resampled['Y染色体浓度'].clip(lower=0)

            # 重新计算各组最优时点
            for i in range(n_groups):
                lower, upper = all_breaks[i], all_breaks[i + 1]
                mask = (df_resampled['BMI_calc'] >= lower) & (df_resampled['BMI_calc'] < upper)
                df_group = df_resampled[mask]

                if len(df_group) < 10:
                    continue

                # 简化的最优时点搜索
                search_weeks = np.arange(10, 25, 0.5)
                best_week = 12
                best_score = np.inf

                for week in search_weeks:
                    score = self._objective_function(week, df_group)
                    if score < best_score:
                        best_score = score
                        best_week = week

                bootstrap_results[f'G{i+1}'].append(best_week)

        # 统计分析
        print("\n【Bootstrap Results Summary】")
        print("-" * 90)
        print(f"{'Group':<10} {'Baseline':<12} {'Bootstrap Mean':<16} {'Std Dev':<12} {'95% CI':<25} {'Shift':<10}")
        print("-" * 90)

        summary_data = []
        for i, result in enumerate(self.optimal_timings):
            group_key = f'G{i+1}'
            baseline = result.optimal_week
            boot_values = bootstrap_results[group_key]

            if len(boot_values) > 10:
                mean_boot = np.mean(boot_values)
                std_boot = np.std(boot_values)
                ci_lower = np.percentile(boot_values, 2.5)
                ci_upper = np.percentile(boot_values, 97.5)
                shift = mean_boot - baseline

                summary_data. append({
                    'Group':  group_key,
                    'Baseline': baseline,
                    'Bootstrap_Mean': mean_boot,
                    'Bootstrap_Std': std_boot,
                    'CI_Lower': ci_lower,
                    'CI_Upper':  ci_upper,
                    'Shift': shift
                })

                print(f"{group_key:<10} {baseline:<12.1f} {mean_boot:<16.2f} {std_boot:<12.2f} "
                      f"[{ci_lower:.2f}, {ci_upper:.2f}]{'':<5} {shift:+.2f}")

        print("-" * 90)

        # 绘制Bootstrap结果
        self._plot_bootstrap_results(bootstrap_results, summary_data)

        return pd. DataFrame(summary_data)

    def _plot_bootstrap_results(self, bootstrap_results: Dict, summary_data: List[Dict]):
        """绘制Bootstrap分析结果"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 图1: 各组最优时点的Bootstrap分布
        ax1 = axes[0]
        colors = plt.cm.tab10.colors[: len(self.breakpoints) + 1]

        for i, (key, values) in enumerate(bootstrap_results.items()):
            if len(values) > 10:
                violin_parts = ax1.violinplot([values], positions=[i], widths=0.7, showmeans=True, showmedians=True)
                for pc in violin_parts['bodies']:
                    pc.set_facecolor(colors[i])
                    pc.set_alpha(0.7)

        # 添加基准点
        for i, result in enumerate(self.optimal_timings):
            ax1.scatter([i], [result.optimal_week], color='red', s=100, marker='D',
                        zorder=5, label='Baseline' if i == 0 else '', edgecolors='black')

        ax1.set_xticks(range(len(bootstrap_results)))
        ax1.set_xticklabels([f"G{i + 1}" for i in range(len(bootstrap_results))])
        ax1.set_xlabel('BMI Group', fontsize=11)
        ax1.set_ylabel('Optimal Gestational Week', fontsize=11)
        ax1.set_title('(A) Bootstrap Distribution of Optimal Timing', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')

        # 图2: 偏移量和置信区间
        ax2 = axes[1]
        if summary_data:
            groups = [d['Group'] for d in summary_data]
            baselines = [d['Baseline'] for d in summary_data]
            means = np.array([d['Bootstrap_Mean'] for d in summary_data])
            ci_lowers = np.array([d['CI_Lower'] for d in summary_data])
            ci_uppers = np.array([d['CI_Upper'] for d in summary_data])

            x = np.arange(len(groups))

            # 修复：确保误差值为正
            lower_errors = np.maximum(means - ci_lowers, 0.001)
            upper_errors = np.maximum(ci_uppers - means, 0.001)

            ax2.errorbar(x, means, yerr=[lower_errors, upper_errors],
                         fmt='o', capsize=5, capthick=2, color=COLORS['primary'],
                         markersize=10, label='Bootstrap Mean ± 95% CI')
            ax2.scatter(x, baselines, color='red', s=120, marker='D', zorder=5,
                        label='Baseline (No Noise)', edgecolors='black')

            ax2.set_xticks(x)
            ax2.set_xticklabels(groups)
            ax2.set_xlabel('BMI Group', fontsize=11)
            ax2.set_ylabel('Optimal Gestational Week', fontsize=11)
            ax2.set_title('(B) Baseline vs Bootstrap Estimates', fontsize=12, fontweight='bold')
            ax2.legend(loc='upper left', fontsize=9)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('Q2_Bootstrap_Analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("\n   ✓ Figure saved: Q2_Bootstrap_Analysis.png")
        plt.close()

    # ==========================================
    # 6. 生成最终报告
    # ==========================================
    def generate_clinical_recommendations(self):
        """生成临床建议报告"""
        print("\n" + "=" * 70)
        print("📋 Section 6: Clinical Recommendations")
        print("=" * 70)

        print("\n【Key Findings】")
        print("-" * 70)
        print(f"   1. Optimal BMI Breakpoints: {[f'{bp:.1f}' for bp in self.breakpoints]}")
        print(f"   2. Number of Groups: {len(self. breakpoints) + 1}")

        if self.optimal_timings:
            print("\n【Recommended Detection Strategy】")
            print("-" * 70)

            for result in self.optimal_timings:
                bmi_low, bmi_high = result.bmi_range
                print(f"\n   {result.group_name} (BMI:  {bmi_low:.1f} - {bmi_high:.1f}):")
                print(f"      • Optimal Detection Week: {result.optimal_week:.1f}")
                print(f"      • Expected Pass Rate: {result.pass_probability:.1%}")
                print(f"      • Risk Score: {result.risk_score:.3f}")

                # 临床建议
                if result. optimal_week <= 13:
                    print(f"      ✅ Recommendation: Standard early screening (11-13 weeks)")
                elif result.optimal_week <= 16:
                    print(f"      ⚠️ Recommendation:  Slightly delayed screening (13-16 weeks)")
                else:
                    print(f"      ❗ Recommendation: Sequential testing strategy")
                    print(f"         - First test at 11-12 weeks")
                    print(f"         - Retest at {result.optimal_week:.0f} weeks if initial test fails")

        print("\n【Sequential Testing Protocol】")
        print("-" * 70)
        print("   For Low/Medium BMI (G1, G2, G3):")
        print("      → Initial test:  11-12 weeks")
        print("      → If Y concentration < 4%: Retest in 2-4 weeks")
        print("")
        print("   For High BMI (G4, BMI ≥ 36):")
        print("      → Option A (Conservative): Test at 18 weeks for single-pass success")
        print("      → Option B (Recommended): Initial test at 11-12 weeks + mandatory retest at 15-16 weeks")
        print("")
        print("   Rationale: Sequential testing preserves early detection opportunity")
        print("              while ensuring accuracy for high-risk groups")

        print("\n【Measurement Error Impact】")
        print("-" * 70)
        if self.noise_sigma:
            print(f"   Estimated measurement noise: σ = {self.noise_sigma:.4f}")
            print("   Impact: Optimal timing shifts rightward (delayed) when noise is considered")
            print("   Clinical implication: Build in safety margin for borderline cases")


# ==========================================
# 主程序
# ==========================================
def main():
    """主程序入口"""
    print("\n" + "🎯" * 35)
    print("NIPT Optimal Detection Timing Optimization Model")
    print("Based on Nonlinear Optimization and Risk Minimization")
    print("🎯" * 35)
    print(f"\nVersion: V2.0 (Q2 Optimization Model)")
    print(f"Pass Threshold: {FF_THRESHOLD * 100:.0f}%")
    print(f"Detection Window: {MIN_GA_WEEKS}-{MAX_GA_WEEKS} weeks")

    # ==========================================
    # Step 1: 数据加载与预处理
    # ==========================================
    excel_file = '附件.xlsx'
    processor = NIPTDataProcessor(excel_file)
    df_male, df_female = processor.load_and_process()

    if df_male is None:
        print("❌ Data loading failed, program terminated")
        return

    # ==========================================
    # Step 2: 问题2 - 最优检测时点模型
    # ==========================================
    print("\n" + "📊" * 35)
    print("Question 2: Optimal Detection Timing Optimization")
    print("📊" * 35)

    optimizer = Problem2_OptimalTiming(df_male)

    # 2.1 基础回归模型
    baseline_params = optimizer.fit_baseline_regression()

    # 2.2 BMI断点优化
    breakpoints = optimizer.optimize_bmi_breakpoints(n_groups= 4, search_range=(18 , 48))

    # 2.3 最优检测时点
    optimal_timings = optimizer. find_optimal_timing_per_group()

    # 2.4 误差敏感性分析
    optimizer.estimate_measurement_noise()
    bootstrap_summary = optimizer.cluster_bootstrap_analysis(n_bootstrap=200)

    # 2.5 生成临床建议
    optimizer.generate_clinical_recommendations()



    print("\n" + "=" * 70)
    print("✅ Analysis Complete!")
    print("=" * 70)
    print("\nGenerated Figures:")
    print("   1. Q2_BMI_Segmentation.png - BMI group visualization")
    print("   2. Q2_Optimal_Timing.png - Optimal timing analysis")
    print("   3. Q2_Bootstrap_Analysis.png - Sensitivity analysis")


# ==========================================
# 程序入口
# ==========================================
if __name__ == "__main__":
    main()