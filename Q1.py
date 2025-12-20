import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Tuple, Dict, List, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

# 统计与机器学习库
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.neural_network import MLPRegressor
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    classification_report, roc_curve, auc,
    precision_recall_curve, confusion_matrix, roc_auc_score,
    r2_score, mean_squared_error, silhouette_score
)

# 统计建模
from scipy import stats
from scipy.optimize import minimize_scalar, minimize
from scipy.interpolate import interp1d
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

# ==========================================
# 全局配置 (Global Configuration)
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong', 'DejaVu Sans', 'Arial Unicode MS',
                                   'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (10, 6)  # 调整单张图的尺寸
plt.rcParams['font.size'] = 11
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
warnings.filterwarnings('ignore')

# 医学常数 (Clinical Constants)
FF_THRESHOLD = 0.04  # 胎儿分数达标阈值 (4%)
MIN_GA_WEEKS = 10  # 最早可检测孕周
MAX_GA_WEEKS = 28  # 最晚推荐检测孕周
CRITICAL_GA_WEEKS = 20  # 临床关键决策时间点

# BMI分类标准 (WHO标准)
BMI_CATEGORIES = {
    'underweight': (0, 18.5),
    'normal': (18.5, 24),
    'overweight': (24, 28),
    'obese_I': (28, 32),
    'obese_II': (32, 36),
    'obese_III': (36, 100)
}

# 颜色方案
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#28A745',
    'warning': '#FFC107',
    'danger': '#DC3545',
    'info': '#17A2B8',
    'dark': '#343A40',
    'light': '#F8F9FA'
}

# 风险权重参数
RISK_WEIGHTS = {
    'failure': 100000,  # 检测失败成本
    'delay_base': 10,  # 延迟基础成本
    'delay_exp': 0.01,  # 延迟指数增长系数
    'uncertainty': 5  # 不确定性惩罚
}


# ==========================================
# 数据类定义 (Data Classes)
# ==========================================
@dataclass
class ModelResults:
    """模型结果数据类"""
    model_name: str
    coefficients: Dict
    r_squared: float
    adj_r_squared: float
    aic: float
    bic: float
    residuals: np.ndarray
    predictions: np.ndarray
    confidence_intervals: pd.DataFrame


@dataclass
class OptimizationResult:
    """优化结果数据类"""
    bmi_group: str
    bmi_range: Tuple[float, float]
    recommended_week: int
    expected_pass_rate: float
    risk_score: float
    data_support: str


# ==========================================
# 模块1: 数据预处理 (Data Preprocessing)
# ==========================================
class NIPTDataProcessor:
    """
    NIPT数据预处理类

    功能:
    - 孕周字符串解析 (如 "12w+3d" → 12.43)
    - 缺失值MICE多重插补
    - BMI计算与验证
    - 特征工程
    - 数据质量报告生成

    Attributes:
        excel_path:  Excel文件路径
        imputer:  MICE插补器
        processing_log: 处理日志
    """

    def __init__(self, excel_path: str):
        """
        初始化数据处理器

        Args:
            excel_path: Excel数据文件路径
        """
        self.excel_path = excel_path
        self.imputer = IterativeImputer(
            max_iter=20,
            random_state=2025,
            min_value=0,
            initial_strategy='median'
        )
        self.processing_log = []
        self.data_quality_report = {}

    def _parse_gestational_age(self, ga_str) -> float:
        """
        解析孕周字符串为数值 (周)

        支持格式:
        - "12w+3d" / "12W+3D" → 12.43
        - "12w3" → 12.43
        - "12. 5" → 12.5
        - 12 → 12.0

        Args:
            ga_str: 孕周字符串或数值

        Returns:
            float: 孕周数值
        """
        if pd.isna(ga_str):
            return np.nan
        try:
            ga_str = str(ga_str).lower().strip().replace('d', '')
            if 'w' in ga_str:
                parts = ga_str.split('w')
                weeks = float(parts[0])
                days = 0
                if len(parts) > 1:
                    day_part = parts[1].replace('+', '').strip()
                    if day_part:
                        days = float(day_part)
                return weeks + days / 7.0
            return float(ga_str)
        except (ValueError, TypeError):
            return np.nan

    def _validate_bmi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        验证BMI合理性 (生物学约束:  15 < BMI < 60)

        Args:
            df: 数据框

        Returns:
            pd. DataFrame: 验证后的数据框
        """
        df = df.copy()
        invalid_mask = (df['BMI_calc'] < 15) | (df['BMI_calc'] > 60)
        n_invalid = invalid_mask.sum()
        if n_invalid > 0:
            self.processing_log.append(f"   ⚠ Found {n_invalid} abnormal BMI values (out of range), marked as missing")
            df.loc[invalid_mask, 'BMI_calc'] = np.nan
        return df

    def _generate_quality_report(self, df: pd.DataFrame, name: str) -> Dict:
        """生成数据质量报告"""
        report = {
            'total_samples': len(df),
            'missing_rates': {},
            'value_ranges': {},
            'outliers': {}
        }

        key_cols = ['Y染色体浓度', 'GA_numeric', 'BMI_calc', '年龄']
        for col in key_cols:
            if col in df.columns:
                report['missing_rates'][col] = df[col].isna().mean()
                report['value_ranges'][col] = (df[col].min(), df[col].max())
                # 检测异常值 (IQR方法)
                Q1, Q3 = df[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
                report['outliers'][col] = outliers

        self.data_quality_report[name] = report
        return report

    def process_dataset(self, df_raw: pd.DataFrame, dataset_name: str = 'data') -> pd.DataFrame:
        """
        核心数据处理流程

        Args:
            df_raw: 原始数据框
            dataset_name: 数据集名称

        Returns:
            pd.DataFrame: 处理后的数据框
        """
        df = df_raw.copy()
        self.processing_log.append(f"\n{'=' * 60}")
        self.processing_log.append(f"📊 Processing dataset: {dataset_name}")
        self.processing_log.append(f"   Original sample size: {len(df)}")

        # 1. 孕周数值化
        if '检测孕周' in df.columns:
            df['GA_numeric'] = df['检测孕周'].apply(self._parse_gestational_age)
            valid_ga = df['GA_numeric'].notna().sum()
            self.processing_log.append(f"   ✓ Gestational age parsing successful: {valid_ga}/{len(df)}")

        # 2. 数值列清洗
        numeric_cols = ['年龄', '身高', '体重']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 3.  MICE多重插补
        available_cols = [c for c in numeric_cols if c in df.columns]
        if available_cols:
            n_missing_before = df[available_cols].isna().sum().sum()
            if n_missing_before > 0:
                df[available_cols] = self.imputer.fit_transform(df[available_cols])
                self.processing_log.append(f"   ✓ MICE imputation: filled {n_missing_before} missing values")

        # 4. 身高单位统一 (cm)
        if '身高' in df.columns:
            if df['身高'].median() < 3:
                df['身高'] = df['身高'] * 100
                self.processing_log.append("   ✓ Height unit conversion: m → cm")

        # 5. BMI计算与验证
        if '身高' in df.columns and '体重' in df.columns:
            df['BMI_calc'] = df['体重'] / ((df['身高'] / 100) ** 2)
            df = self._validate_bmi(df)
            self.processing_log.append(f"   ✓ BMI calculation complete: {df['BMI_calc'].notna().sum()} valid values")

        # 6. 特征工程
        if 'BMI_calc' in df.columns:
            df['Log_BMI'] = np.log(df['BMI_calc'] + 1e-5)
            df['BMI_squared'] = df['BMI_calc'] ** 2

        if 'GA_numeric' in df.columns:
            df['GA_squared'] = df['GA_numeric'] ** 2

        if 'GA_numeric' in df.columns and 'BMI_calc' in df.columns:
            # 稀释效应交互项
            df['GA_BMI_ratio'] = df['GA_numeric'] / (df['BMI_calc'] + 1e-5)
            df['GA_BMI_interaction'] = df['GA_numeric'] * df['BMI_calc']

        # 7. IVF编码
        if 'IVF妊娠' in df.columns:
            df['Is_IVF'] = df['IVF妊娠'].apply(lambda x: 1 if str(x).strip() == '是' else 0)
        else:
            df['Is_IVF'] = 0

        # 8. 标签处理
        if '胎儿是否健康' in df.columns:
            df['Target_Label'] = df['胎儿是否健康'].apply(
                lambda x: 0 if str(x).strip() == '是' else 1
            )

        # 9. 达标标记
        if 'Y染色体浓度' in df.columns:
            df['FF_Pass'] = (df['Y染色体浓度'] >= FF_THRESHOLD).astype(int)

        # 生成质量报告
        self._generate_quality_report(df, dataset_name)

        self.processing_log.append(f"   ✓ Final sample size: {len(df)}")
        return df

    def load_and_process(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        加载并处理Excel数据

        Returns:
            Tuple:  (男胎数据, 女胎数据)
        """
        print("=" * 70)
        print("📂 NIPT Data Loading and Preprocessing")
        print("=" * 70)

        try:
            # 读取数据
            df_male = pd.read_excel(self.excel_path, sheet_name='男胎检测数据')
            df_female = pd.read_excel(self.excel_path, sheet_name='女胎检测数据')

            # 处理数据
            df_male_processed = self.process_dataset(df_male, 'Male Fetus')
            df_female_processed = self.process_dataset(df_female, 'Female Fetus')

            # 输出处理日志
            for log in self.processing_log:
                print(log)

            # 数据概览
            print(f"\n📊 Data Quality Overview:")
            print(f"   Male fetus samples:  {len(df_male_processed)}")
            print(f"   Female fetus samples: {len(df_female_processed)}")

            if 'BMI_calc' in df_male_processed.columns:
                print(f"\n   Male Fetus BMI Distribution:")
                print(
                    f"      Range: {df_male_processed['BMI_calc'].min():.1f} - {df_male_processed['BMI_calc'].max():.1f}")
                print(
                    f"      Mean±SD: {df_male_processed['BMI_calc'].mean():.1f}±{df_male_processed['BMI_calc'].std():.1f}")
                print(
                    f"      Median(IQR): {df_male_processed['BMI_calc'].median():.1f} ({df_male_processed['BMI_calc'].quantile(0.25):.1f}-{df_male_processed['BMI_calc'].quantile(0.75):.1f})")

            if 'Y染色体浓度' in df_male_processed.columns:
                pass_rate = (df_male_processed['Y染色体浓度'] >= FF_THRESHOLD).mean()
                print(f"\n   Y Chromosome Concentration Pass Rate: {pass_rate:.1%}")

            return df_male_processed, df_female_processed

        except FileNotFoundError:
            print(f"❌ File not found: {self.excel_path}")
            return None, None
        except Exception as e:
            print(f"❌ Loading error: {e}")
            import traceback
            traceback.print_exc()
            return None, None


# ==========================================
# 模块2: 问题1 - 相关性分析与回归建模
# ==========================================
class Problem1_CorrelationRegression:
    """
    问题1:  Y染色体浓度与孕周、BMI等指标的相关性分析与回归建模

    方法论:
    1. Pearson/Spearman相关性分析
    2. 多元线性回归 (OLS) + VIF检验
    3. 多项式回归 (二次项捕捉非线性)
    4. 线性混合效应模型 (LMM) - 处理纵向数据
    5. 完整的模型诊断 (残差分析、正态性检验、异方差检验)

    医学背景:
    - 胎儿分数(FF)随孕周增加而上升 (胎盘发育促进cffDNA释放)
    - FF随BMI增加而下降 (母体血容量增大导致稀释效应)
    - 年龄对FF的影响相对较小
    """

    def __init__(self):
        self.linear_model = None
        self.poly_model = None
        self.lmm_model = None
        self.correlation_results = None
        self.model_comparison = None
        self.coefficients = {}

    def correlation_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        相关性分析 (Pearson + Spearman + 偏相关)

        Args:
            df:  数据框

        Returns:
            pd.DataFrame: 用于后续分析的清洗数据
        """
        print("\n" + "=" * 70)
        print("📊 Question 1 - Step 1: Correlation Analysis")
        print("=" * 70)

        # 创建中文到英文的列名映射
        column_mapping = {
            'Y染色体浓度': 'Y_Concentration',
            'GA_numeric': 'Gestational_Age',
            'BMI_calc': 'BMI',
            '年龄': 'Age',
            '体重': 'Weight',
            '身高': 'Height'
        }

        # 创建英文到中文的反向映射 (用于数据访问)
        reverse_mapping = {v: k for k, v in column_mapping.items()}

        variables = ['Y染色体浓度', 'GA_numeric', 'BMI_calc', '年龄', '体重', '身高']
        available_vars = [v for v in variables if v in df.columns]
        df_corr = df[available_vars].dropna()

        print(f"\nValid sample size: n = {len(df_corr)}")

        # 描述性统计 - 使用英文列名
        print("\n【Descriptive Statistics】")
        print("-" * 75)
        print(f"{'Variable':<18} {'Mean': >10} {'Std Dev':>10} {'Min':>10} {'Median':>10} {'Max':>10}")
        print("-" * 75)
        for var in available_vars:
            eng_name = column_mapping.get(var, var)
            print(f"{eng_name:<18} {df_corr[var].mean():>10.4f} {df_corr[var].std():>10.4f} "
                  f"{df_corr[var].min():>10.4f} {df_corr[var].median():>10.4f} {df_corr[var].max():>10.4f}")
        print("-" * 75)

        # 相关性检验 - 使用英文变量名
        results = []
        print("\n【Correlation Test】")
        print("-" * 85)
        print(f"{'Variable':<18} {'Pearson r': >10} {'95% CI':>20} {'p-value':>12} {'Spearman ρ':>12} {'Sig': >8}")
        print("-" * 85)

        for var in available_vars[1:]:
            # Pearson相关
            r_pearson, p_pearson = stats.pearsonr(df_corr['Y染色体浓度'], df_corr[var])

            # Pearson相关系数的置信区间 (Fisher z变换)
            n = len(df_corr)
            z = np.arctanh(r_pearson)
            se = 1 / np.sqrt(n - 3)
            z_lower, z_upper = z - 1.96 * se, z + 1.96 * se
            r_lower, r_upper = np.tanh(z_lower), np.tanh(z_upper)

            # Spearman相关
            r_spearman, p_spearman = stats.spearmanr(df_corr['Y染色体浓度'], df_corr[var])

            # 效应量判断 (Cohen's guidelines)
            if abs(r_pearson) < 0.1:
                effect_size = 'Negligible'
            elif abs(r_pearson) < 0.3:
                effect_size = 'Small'
            elif abs(r_pearson) < 0.5:
                effect_size = 'Medium'
            else:
                effect_size = 'Large'

            sig = '***' if p_pearson < 0.001 else '**' if p_pearson < 0.01 else '*' if p_pearson < 0.05 else 'ns'

            eng_var_name = column_mapping.get(var, var)
            results.append({
                'Variable': eng_var_name,
                'Pearson_r': r_pearson,
                'CI_lower': r_lower,
                'CI_upper': r_upper,
                'Pearson_p': p_pearson,
                'Spearman_rho': r_spearman,
                'Spearman_p': p_spearman,
                'Effect_Size': effect_size,
                'Significance': sig
            })

            ci_str = f"[{r_lower:.3f}, {r_upper:.3f}]"
            print(
                f"{eng_var_name: <18} {r_pearson:>10.4f} {ci_str:>20} {p_pearson:>12.2e} {r_spearman:>12.4f} {sig:>8}")

        print("-" * 85)
        print("Significance level:  *** p<0.001, ** p<0.01, * p<0.05, ns not significant")

        # 医学意义解读
        print("\n📌 Medical Interpretation:")
        print("   • Gestational Age (GA) - Positive Correlation:  Placental development promotes cffDNA release")
        print("   • BMI - Negative Correlation: Increased maternal blood volume causes dilution effect")
        print(
            "   • Weight - Negative Correlation: Similar mechanism to BMI, adipose tissue releases more maternal cfDNA")
        print("   • These findings are consistent with ACOG clinical guidelines")

        self.correlation_results = pd.DataFrame(results)

        # 绘制相关性分析图（拆分为独立图片）
        self._plot_correlation_analysis(df_corr, available_vars, column_mapping)

        return df_corr

    def _plot_correlation_analysis(self, df: pd.DataFrame, variables: List[str], column_mapping: Dict[str, str]):
        """绘制相关性分析图（拆分为独立的四张图片）"""

        # 1. 相关性热力图
        plt.figure(figsize=(10, 8))
        corr_matrix = df[variables].corr()
        # 创建英文标签
        english_labels = [column_mapping.get(var, var) for var in variables]
        corr_matrix.index = english_labels
        corr_matrix.columns = english_labels

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(corr_matrix, mask=False, annot=True, fmt='.3f',
                    cmap='RdBu_r', center=0,
                    square=True, linewidths=0.5,
                    cbar_kws={'shrink': 0.8, 'label': 'Pearson r'},
                    annot_kws={'size': 10})
        plt.title('(A) Correlation Matrix of Variables', fontsize=12, fontweight='bold', pad=10)
        plt.tight_layout()
        plt.savefig('Q1_correlation_matrix.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("   ✓ Figure saved: Q1_correlation_matrix.png")
        plt.close()

        # 2. Y浓度 vs 孕周
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(df['GA_numeric'], df['Y染色体浓度'],
                              c=df['BMI_calc'], cmap='viridis', alpha=0.5, s=20, edgecolors='none')
        # 添加趋势线
        z = np.polyfit(df['GA_numeric'], df['Y染色体浓度'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df['GA_numeric'].min(), df['GA_numeric'].max(), 100)
        plt.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Linear Trend: y={z[0]:.4f}x+{z[1]:.4f}')
        plt.axhline(y=FF_THRESHOLD, color='green', linestyle=':', linewidth=2,
                    label=f'Pass Threshold ({FF_THRESHOLD * 100:.0f}%)')
        plt.xlabel('Gestational Age (weeks)', fontsize=11)
        plt.ylabel('Y Chromosome Concentration', fontsize=11)
        plt.title('(B) Y Concentration vs Gestational Age (Color=BMI)', fontsize=12, fontweight='bold', pad=10)
        plt.legend(loc='upper left', fontsize=9)
        cbar = plt.colorbar(scatter)
        cbar.set_label('BMI (kg/m²)', fontsize=10)
        plt.tight_layout()
        plt.savefig('Q1_y_concentration_vs_ga.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("   ✓ Figure saved: Q1_y_concentration_vs_ga.png")
        plt.close()

        # 3. Y浓度 vs BMI
        plt.figure(figsize=(10, 6))
        scatter3 = plt.scatter(df['BMI_calc'], df['Y染色体浓度'],
                               c=df['GA_numeric'], cmap='plasma', alpha=0.5, s=20, edgecolors='none')
        z2 = np.polyfit(df['BMI_calc'], df['Y染色体浓度'], 1)
        p2 = np.poly1d(z2)
        x_line2 = np.linspace(df['BMI_calc'].min(), df['BMI_calc'].max(), 100)
        plt.plot(x_line2, p2(x_line2), 'r--', linewidth=2, label=f'Linear Trend: y={z2[0]:.4f}x+{z2[1]:.4f}')
        plt.axhline(y=FF_THRESHOLD, color='green', linestyle=':', linewidth=2)
        plt.xlabel('BMI (kg/m²)', fontsize=11)
        plt.ylabel('Y Chromosome Concentration', fontsize=11)
        plt.title('(C) Y Concentration vs BMI (Color=GA)', fontsize=12, fontweight='bold', pad=10)
        plt.legend(loc='upper right', fontsize=9)
        cbar3 = plt.colorbar(scatter3)
        cbar3.set_label('Gestational Age (weeks)', fontsize=10)
        plt.tight_layout()
        plt.savefig('Q1_y_concentration_vs_bmi.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("   ✓ Figure saved: Q1_y_concentration_vs_bmi.png")
        plt.close()

        # 4. 达标率随BMI变化
        plt.figure(figsize=(10, 6))
        bmi_bins = np.arange(18, 48, 2)
        df['BMI_bin'] = pd.cut(df['BMI_calc'], bins=bmi_bins)
        pass_rates = df.groupby('BMI_bin')['Y染色体浓度'].apply(lambda x: (x >= FF_THRESHOLD).mean())
        sample_sizes = df.groupby('BMI_bin').size()

        valid_idx = sample_sizes >= 5
        x_plot = [(interval.left + interval.right) / 2 for interval in pass_rates.index[valid_idx]]
        y_plot = pass_rates.values[valid_idx]

        plt.bar(x_plot, y_plot, width=1.8, alpha=0.7, color=COLORS['primary'], edgecolor='black')
        plt.axhline(y=0.85, color='orange', linestyle='--', linewidth=2, label='85% Threshold')
        plt.axhline(y=0.95, color='green', linestyle='--', linewidth=2, label='95% Threshold')
        plt.xlabel('BMI (kg/m²)', fontsize=11)
        plt.ylabel('Pass Rate', fontsize=11)
        plt.title('(D) Pass Rate by BMI Group', fontsize=12, fontweight='bold', pad=10)
        plt.legend(loc='lower left', fontsize=9)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig('Q1_pass_rate_by_bmi.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("   ✓ Figure saved: Q1_pass_rate_by_bmi.png")
        plt.close()

    def linear_regression(self, df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
        """
        多元线性回归模型 (OLS)

        Args:
            df:  数据框

        Returns:
            statsmodels回归结果对象
        """
        print("\n" + "=" * 70)
        print("📊 Question 1 - Step 2: Multiple Linear Regression Model (OLS)")
        print("=" * 70)

        feature_cols = ['GA_numeric', 'BMI_calc', '年龄']
        available_features = [f for f in feature_cols if f in df.columns]
        df_model = df[available_features + ['Y染色体浓度']].dropna()

        X = df_model[available_features]
        y = df_model['Y染色体浓度']

        # OLS回归
        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit()

        # 模型摘要
        print("\n【Model Summary】")
        print(f"   Sample size: n = {len(df_model)}")
        print(f"   R² = {model.rsquared:.4f}")
        print(f"   Adjusted R² = {model.rsquared_adj:.4f}")
        print(f"   F-statistic = {model.fvalue:.2f} (p = {model.f_pvalue:.2e})")
        print(f"   AIC = {model.aic:.2f}")
        print(f"   BIC = {model.bic:.2f}")
        print(f"   Residual Std Error = {np.sqrt(model.mse_resid):.6f}")

        # 回归系数表
        print("\n【Regression Coefficients】")
        print("-" * 85)
        print(
            f"{'Variable':<15} {'Coefficient β': >12} {'Std Error':>10} {'t value':>10} {'p-value':>12} {'95% CI': >25}")
        print("-" * 85)

        # 创建英文变量名映射
        eng_var_names = ['Intercept']
        for feat in available_features:
            if feat == 'GA_numeric':
                eng_var_names.append('Gestational_Age')
            elif feat == 'BMI_calc':
                eng_var_names.append('BMI')
            elif feat == '年龄':
                eng_var_names.append('Age')
            else:
                eng_var_names.append(feat)

        conf_int = model.conf_int()
        for i, name in enumerate(eng_var_names):
            ci = f"[{conf_int.iloc[i, 0]:.6f}, {conf_int.iloc[i, 1]:.6f}]"
            sig = '***' if model.pvalues[i] < 0.001 else '**' if model.pvalues[i] < 0.01 else '*' if model.pvalues[
                                                                                                         i] < 0.05 else ''
            print(f"{name:<15} {model.params[i]: >12.6f} {model.bse[i]:>10.6f} "
                  f"{model.tvalues[i]:>10.3f} {model.pvalues[i]:>12.2e} {ci: >25} {sig}")
        print("-" * 85)

        # 保存关键系数
        self.coefficients = {
            'intercept': model.params[0],
            'GA': model.params[1] if 'GA_numeric' in available_features else 0,
            'BMI': model.params[available_features.index('BMI_calc') + 1] if 'BMI_calc' in available_features else 0,
            'residual_std': np.sqrt(model.mse_resid)
        }

        # VIF多重共线性检验
        print("\n【Multicollinearity Test (VIF)】")
        print("-" * 50)
        vif_data = []
        for i, col in enumerate(available_features):
            vif = variance_inflation_factor(X.values, i)
            if vif < 5:
                status = '✓ No multicollinearity'
            elif vif < 10:
                status = '⚠ Moderate multicollinearity'
            else:
                status = '❌ Severe multicollinearity'

            # 转换为英文变量名
            if col == 'GA_numeric':
                eng_name = 'Gestational_Age'
            elif col == 'BMI_calc':
                eng_name = 'BMI'
            elif col == '年龄':
                eng_name = 'Age'
            else:
                eng_name = col

            print(f"   {eng_name:<15} VIF = {vif:.2f} {status}")
            vif_data.append({'Variable': eng_name, 'VIF': vif, 'Status': status})
        print("-" * 50)
        print("   Standard:  VIF < 5 no multicollinearity; 5-10 moderate; >10 severe")

        # 残差诊断（拆分为独立图片）
        self._residual_diagnostics(model, 'Linear', df_model, available_features)

        self.linear_model = model
        return model

    def polynomial_regression(self, df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
        """
        多项式回归 (捕捉非线性效应)

        Args:
            df: 数据框

        Returns:
            statsmodels回归结果对象
        """
        print("\n" + "=" * 70)
        print("📊 Question 1 - Step 3: Polynomial Regression Model (Quadratic)")
        print("=" * 70)

        feature_cols = ['GA_numeric', 'BMI_calc']
        available_features = [f for f in feature_cols if f in df.columns]
        df_model = df[available_features + ['Y染色体浓度']].dropna()

        X = df_model[available_features].values
        y = df_model['Y染色体浓度'].values

        # 二次多项式特征
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        feature_names = poly.get_feature_names_out(available_features)

        # 转换特征名为英文
        english_feature_names = []
        for fname in feature_names:
            eng_fname = fname.replace('GA_numeric', 'Gestational_Age').replace('BMI_calc', 'BMI').replace('年龄', 'Age')
            english_feature_names.append(eng_fname)

        X_poly_sm = sm.add_constant(X_poly)
        model_poly = sm.OLS(y, X_poly_sm).fit()

        print(f"\n【Polynomial Regression Summary】")
        print(f"   R² = {model_poly.rsquared:.4f}")
        print(f"   Adjusted R² = {model_poly.rsquared_adj:.4f}")
        print(f"   AIC = {model_poly.aic:.2f}")

        print("\n【Polynomial Regression Coefficients】")
        print("-" * 70)
        print(f"{'Feature':<30} {'Coefficient': >12} {'p-value':>15}")
        print("-" * 70)
        for i, name in enumerate(['Intercept'] + english_feature_names):
            sig = '***' if model_poly.pvalues[i] < 0.001 else '**' if model_poly.pvalues[i] < 0.01 else '*' if \
                model_poly.pvalues[i] < 0.05 else ''
            print(f"{name:<30} {model_poly.params[i]:>12.6f} {model_poly.pvalues[i]: >15.4e} {sig}")
        print("-" * 70)

        # 模型对比
        print("\n【Model Comparison】")
        print("-" * 60)
        print(f"{'Metric':<20} {'Linear Model':>15} {'Polynomial Model': >15}")
        print("-" * 60)
        print(f"{'R²':<20} {self.linear_model.rsquared: >15.4f} {model_poly.rsquared:>15.4f}")
        print(f"{'Adjusted R²':<20} {self.linear_model.rsquared_adj:>15.4f} {model_poly.rsquared_adj:>15.4f}")
        print(f"{'AIC': <20} {self.linear_model.aic:>15.2f} {model_poly.aic:>15.2f}")
        print(f"{'BIC':<20} {self.linear_model.bic:>15.2f} {model_poly.bic:>15.2f}")
        print("-" * 60)

        # 模型选择
        aic_diff = self.linear_model.aic - model_poly.aic
        if aic_diff > 2:
            print(f">>> Recommendation: Polynomial model is better (ΔAIC = {aic_diff:.2f} > 2)")
        elif aic_diff < -2:
            print(f">>> Recommendation: Linear model is better (ΔAIC = {aic_diff:.2f} < -2)")
        else:
            print(f">>> Recommendation: Models are equivalent, linear model is simpler and preferred")

        self.poly_model = model_poly
        return model_poly

    def _residual_diagnostics(self, model, model_name: str, df: pd.DataFrame, features: List[str]):
        """
        残差诊断（拆分为独立的四张图片）

        包含:
        1. 残差 vs 拟合值图
        2. Q-Q图
        3. 残差分布直方图
        4. Scale-Location图
        5. Shapiro-Wilk正态性检验
        6. Breusch-Pagan异方差检验
        """
        residuals = model.resid
        fitted = model.fittedvalues

        # 1. 残差 vs 拟合值
        plt.figure(figsize=(10, 6))
        plt.scatter(fitted, residuals, alpha=0.4, s=15, color=COLORS['primary'], edgecolors='none')
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        # LOWESS平滑
        try:
            lowess = sm.nonparametric.lowess(residuals, fitted, frac=0.3)
            plt.plot(lowess[:, 0], lowess[:, 1], color='orange', linewidth=2, label='LOWESS')
        except:
            pass
        plt.xlabel('Fitted Values', fontsize=11)
        plt.ylabel('Residuals', fontsize=11)
        plt.title(f'{model_name} Model: Residuals vs Fitted Values', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'Q1_{model_name}_residuals_vs_fitted.png', dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✓ Diagnostic plot saved: Q1_{model_name}_residuals_vs_fitted.png")
        plt.close()

        # 2. Q-Q图
        plt.figure(figsize=(10, 6))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title(f'{model_name} Model: Q-Q Plot (Normality Test)', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'Q1_{model_name}_qq_plot.png', dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✓ Diagnostic plot saved: Q1_{model_name}_qq_plot.png")
        plt.close()

        # 3. 残差分布
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=50, density=True, alpha=0.7,
                 edgecolor='black', color=COLORS['primary'], label='Residual Distribution')
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        plt.plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()),
                 'r-', linewidth=2, label='Normal Distribution')
        plt.xlabel('Residuals', fontsize=11)
        plt.ylabel('Density', fontsize=11)
        plt.title(f'{model_name} Model: Residual Distribution', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'Q1_{model_name}_residual_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✓ Diagnostic plot saved: Q1_{model_name}_residual_distribution.png")
        plt.close()

        # 4. Scale-Location图
        plt.figure(figsize=(10, 6))
        sqrt_abs_resid = np.sqrt(np.abs(residuals))
        plt.scatter(fitted, sqrt_abs_resid, alpha=0.4, s=15, color=COLORS['primary'], edgecolors='none')
        try:
            lowess2 = sm.nonparametric.lowess(sqrt_abs_resid, fitted, frac=0.3)
            plt.plot(lowess2[:, 0], lowess2[:, 1], color='orange', linewidth=2, label='LOWESS')
        except:
            pass
        plt.xlabel('Fitted Values', fontsize=11)
        plt.ylabel('√|Residuals|', fontsize=11)
        plt.title(f'{model_name} Model: Scale-Location Plot (Homoscedasticity)', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'Q1_{model_name}_scale_location.png', dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✓ Diagnostic plot saved: Q1_{model_name}_scale_location.png")
        plt.close()

        # 统计检验
        print("\n【Residual Diagnostic Tests】")

        # Shapiro-Wilk正态性检验
        sample_size = min(5000, len(residuals))
        if sample_size < len(residuals):
            resid_sample = np.random.choice(residuals, sample_size, replace=False)
        else:
            resid_sample = residuals
        stat_sw, p_sw = stats.shapiro(resid_sample)
        sw_result = '✓ Normal Distribution' if p_sw > 0.05 else '⚠ Not Normal'
        print(f"   Shapiro-Wilk Test: W = {stat_sw:.4f}, p = {p_sw:.4e} {sw_result}")

        # Breusch-Pagan异方差检验
        try:
            bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, model.model.exog)
            bp_result = '✓ Homoscedastic' if bp_pvalue > 0.05 else '⚠ Heteroscedastic'
            print(f"   Breusch-Pagan Test: χ² = {bp_stat:.4f}, p = {bp_pvalue:.4e} {bp_result}")
        except Exception as e:
            print(f"   Breusch-Pagan Test: Cannot compute ({e})")

        # Durbin-Watson自相关检验
        from statsmodels.stats.stattools import durbin_watson
        dw = durbin_watson(residuals)
        if 1.5 < dw < 2.5:
            dw_result = '✓ No Autocorrelation'
        else:
            dw_result = '⚠ Possible Autocorrelation'
        print(f"   Durbin-Watson Test: DW = {dw:.4f} {dw_result}")

    def generate_final_equation(self):
        """生成最终回归方程及临床解释"""
        print("\n" + "=" * 70)
        print("📌 Question 1 Final Conclusions and Clinical Interpretation")
        print("=" * 70)

        if self.linear_model is None:
            print("❌ Please run linear regression model first")
            return

        coefs = self.linear_model.params
        param_names = self.linear_model.model.exog_names

        # 构建方程
        eq_parts = [f"{coefs[0]:.6f}"]
        for i, name in enumerate(param_names[1:], 1):
            if coefs[i] >= 0:
                eq_parts.append(f"+ {coefs[i]:.6f}×{name}")
            else:
                eq_parts.append(f"- {abs(coefs[i]):.6f}×{name}")

        print("\n【Regression Equation】")
        print(f"   Y Chromosome Concentration = {' '.join(eq_parts)}")

        print("\n【Clinical Interpretation】")
        if 'GA_numeric' in param_names:
            idx = param_names.index('GA_numeric')
            print(f"   • Each additional week of gestation increases Y concentration by {coefs[idx] * 100:.4f}%")

        if 'BMI_calc' in param_names:
            idx = param_names.index('BMI_calc')
            print(f"   • Each unit increase in BMI decreases Y concentration by {-coefs[idx] * 100:.4f}%")

        if 'GA_numeric' in param_names and 'BMI_calc' in param_names:
            ga_idx = param_names.index('GA_numeric')
            bmi_idx = param_names.index('BMI_calc')
            bmi_per_week = abs(coefs[ga_idx] / coefs[bmi_idx])
            print(
                f"   • Clinical Implication: For every {bmi_per_week:.1f} unit increase in BMI, one additional week of gestation is needed")

        print("\n【Model Applicability】")
        print(f"   • Model explains {self.linear_model.rsquared * 100:.1f}% of Y concentration variation")
        print(f"   • Prediction standard error: ±{np.sqrt(self.linear_model.mse_resid) * 100:.2f}%")
        print("   • Suitable for individualized NIPT testing timing recommendations")

        return self.coefficients


class ReportGenerator:
    """
    综合报告生成器

    功能:
    - 生成学术级结果汇总
    - 输出临床建议
    - 保存所有结果到Excel
    """

    def __init__(self):
        self.results = {}

    def add_result(self, key: str, value):
        """添加结果"""
        self.results[key] = value

    def generate_summary(self):
        """生成综合摘要"""
        print("\n" + "=" * 70)
        print("📋 Comprehensive Analysis Report")
        print("=" * 70)

        print("\n【Question 1: Correlation and Regression Analysis】")
        if 'coefficients' in self.results:
            coefs = self.results['coefficients']
            print(f"   • Y concentration increases with gestational age: +{coefs.get('GA', 0) * 100:.4f}%/week")
            print(f"   • Y concentration change with BMI: {coefs.get('BMI', 0) * 100:.4f}%/(kg/m²)")
            print(
                f"   • Clinical significance: Pregnant women with high BMI need delayed testing for detection success")


# ==========================================
# 主程序
# ==========================================
def main():
    """主程序入口"""
    print("\n" + "🎯" * 35)
    print("NIPT Non-Invasive Prenatal Testing Complete Academic Solution")
    print("Non-Invasive Prenatal Testing Complete Academic Solution")
    print("🎯" * 35)
    print(f"\nVersion: V4.0 (Academic Optimization Final)")
    print(f"Pass Threshold: {FF_THRESHOLD * 100:.0f}%")
    print(f"Detection Window: {MIN_GA_WEEKS}-{MAX_GA_WEEKS} weeks")

    # 初始化报告生成器
    report = ReportGenerator()

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
    # Step 2: 问题1 - 相关性分析与回归建模
    # ==========================================
    print("\n" + "📊" * 35)
    print("Question 1: Y Chromosome Concentration Correlation and Regression Analysis")
    print("📊" * 35)

    q1 = Problem1_CorrelationRegression()

    # 相关性分析
    df_corr = q1.correlation_analysis(df_male)
    report.add_result('correlation_results', q1.correlation_results)

    # 线性回归
    q1.linear_regression(df_corr)

    # 多项式回归
    q1.polynomial_regression(df_corr)

    # 生成最终方程
    coefficients = q1.generate_final_equation()
    report.add_result('coefficients', coefficients)


# ==========================================
# 程序入口
# ==========================================
if __name__ == "__main__":
    main()