import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class BankMetrics:
    corporate_interest_income: float  # 公司业务利息收入
    personal_interest_income: float   # 个人业务利息收入
    non_interest_income: float        # 非利息收入
    investment_return: float          # 投资回报

class FuzzyInvestmentAdvisor:
    def __init__(self):
        # 定义模糊集合的参数
        self.low_threshold = 0.3
        self.high_threshold = 0.7
        self.medium_threshold = 0.5

    def _calculate_ratios(self, metrics: BankMetrics) -> dict:
        total_income = (metrics.corporate_interest_income + 
                       metrics.personal_interest_income + 
                       metrics.non_interest_income)
        
        return {
            'corporate_ratio': metrics.corporate_interest_income / total_income,
            'personal_ratio': metrics.personal_interest_income / total_income,
            'non_interest_ratio': metrics.non_interest_income / total_income,
            'return_rate': metrics.investment_return
        }

    def _fuzzy_membership(self, value: float, center: float, spread: float) -> float:
        """计算模糊集合的隶属度"""
        return np.exp(-((value - center) ** 2) / (2 * spread ** 2))

    def _evaluate_cash_investment(self, ratios: dict) -> float:
        """评估现金投资的适合度"""
        score = 0
        # 非利息收入占比高
        score += self._fuzzy_membership(ratios['non_interest_ratio'], 0.8, 0.2)
        # 投资回报率低或不稳定（这里假设低于0.5为低）
        score += self._fuzzy_membership(ratios['return_rate'], 0.3, 0.2)
        # 个人业务利息收入占比高
        score += self._fuzzy_membership(ratios['personal_ratio'], 0.7, 0.2)
        return score / 3

    def _evaluate_fixed_asset_investment(self, ratios: dict) -> float:
        """评估固定资产投资的适合度"""
        score = 0
        # 公司业务利息收入高
        score += self._fuzzy_membership(ratios['corporate_ratio'], 0.8, 0.2)
        # 投资回报率高且稳定
        score += self._fuzzy_membership(ratios['return_rate'], 0.8, 0.2)
        return score / 2

    def _evaluate_mixed_investment(self, ratios: dict) -> float:
        """评估混合投资的适合度"""
        score = 0
        # 公司与个人业务利息收入均衡
        balance = abs(ratios['corporate_ratio'] - ratios['personal_ratio'])
        score += self._fuzzy_membership(balance, 0, 0.2)
        # 投资回报中等偏高
        score += self._fuzzy_membership(ratios['return_rate'], 0.6, 0.2)
        # 非利息收入适中
        score += self._fuzzy_membership(ratios['non_interest_ratio'], 0.5, 0.2)
        return score / 3

    def recommend_investment(self, metrics: BankMetrics) -> Tuple[str, Optional[Tuple[float, float]]]:
        ratios = self._calculate_ratios(metrics)
        
        # 计算每种投资策略的得分
        cash_score = self._evaluate_cash_investment(ratios)
        fixed_score = self._evaluate_fixed_asset_investment(ratios)
        mixed_score = self._evaluate_mixed_investment(ratios)
        
        # 找出得分最高的策略
        scores = [cash_score, fixed_score, mixed_score]
        max_score_idx = np.argmax(scores)
        
        if max_score_idx == 0:
            return "现金投资", None
        elif max_score_idx == 1:
            return "固定资产投资", None
        else:
            # 如果是混合投资，计算两种投资的比例
            total = cash_score + fixed_score
            cash_ratio = cash_score / total
            fixed_ratio = fixed_score / total
            return "混合投资", (fixed_ratio, cash_ratio)

def main():
    # 示例使用
    advisor = FuzzyInvestmentAdvisor()
    
    # 示例数据
    metrics = BankMetrics(
        corporate_interest_income=1000000,
        personal_interest_income=800000,
        non_interest_income=500000,
        investment_return=0.6
    )
    
    investment_type, ratios = advisor.recommend_investment(metrics)
    print(f"推荐的投资策略: {investment_type}")
    if ratios:
        fixed_ratio, cash_ratio = ratios
        print(f"固定资产投资比例: {fixed_ratio:.2%}")
        print(f"现金投资比例: {cash_ratio:.2%}")

if __name__ == "__main__":
    main() 