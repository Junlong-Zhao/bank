import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class MarketMetrics:
    corporate_interest_income: float    # 公司业务利息收入
    personal_interest_income: float     # 个人业务利息收入
    non_interest_income: float          # 非利息收入
    investment_return: float            # 投资回报

class InvestmentTimingAdvisor:
    def __init__(self):
        # Q-learning 参数
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 0.2
        
        # 状态空间离散化参数
        self.state_bins = 5
        
        # 初始化Q表
        self.q_table = {}
        
        # 定义可能的投资时间点（项目开始后的月份）
        self.investment_months = [0, 3, 6, 9, 12]  # 0表示立即投资

    def _normalize_metrics(self, metrics: MarketMetrics) -> list[float]:
        """归一化输入指标"""
        total_income = (metrics.corporate_interest_income + 
                       metrics.personal_interest_income + 
                       metrics.non_interest_income)
        
        return [
            metrics.corporate_interest_income / total_income,
            metrics.personal_interest_income / total_income,
            metrics.non_interest_income / total_income,
            metrics.investment_return
        ]

    def _discretize_state(self, normalized_metrics: list[float]) -> Tuple:
        """将连续状态空间离散化"""
        discretized = []
        for value in normalized_metrics:
            bin_idx = min(int(value * self.state_bins), self.state_bins - 1)
            discretized.append(bin_idx)
        return tuple(discretized)

    def _calculate_market_condition(self, metrics: MarketMetrics) -> float:
        """计算市场状况评分"""
        normalized = self._normalize_metrics(metrics)
        
        # 计算综合得分
        score = 0.0
        
        # 评估公司业务占比
        score += normalized[0] * 0.3  # 30% 权重
        
        # 评估个人业务占比
        score += normalized[1] * 0.2  # 20% 权重
        
        # 评估非利息收入的影响
        score += normalized[2] * 0.2  # 20% 权重
        
        # 评估投资回报
        score += normalized[3] * 0.3  # 30% 权重
        
        return score

    def _evaluate_timing_score(self, market_condition: float, month: int) -> float:
        """评估特定时间点的得分"""
        # 基于项目时间的衰减因子
        time_decay = np.exp(-0.1 * month)  # 时间越长，机会成本越高
        
        # 考虑市场状况和时间衰减
        return market_condition * time_decay

    def recommend_timing(self, metrics: MarketMetrics) -> str:
        """推荐投资时机"""
        normalized_metrics = self._normalize_metrics(metrics)
        state = self._discretize_state(normalized_metrics)
        
        # 计算市场状况
        market_condition = self._calculate_market_condition(metrics)
        
        # 获取当前状态的Q值
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.investment_months))
        
        # ε-贪婪策略
        if np.random.random() < self.exploration_rate:
            month_idx = np.random.randint(len(self.investment_months))
        else:
            month_idx = np.argmax(self.q_table[state])
        
        # 计算奖励并更新Q值
        selected_month = self.investment_months[month_idx]
        reward = self._evaluate_timing_score(market_condition, selected_month)
        
        old_value = self.q_table[state][month_idx]
        next_max = np.max(self.q_table[state])
        
        # Q-learning更新公式
        new_value = (1 - self.learning_rate) * old_value + \
                   self.learning_rate * (reward + self.discount_factor * next_max)
        self.q_table[state][month_idx] = new_value
        
        # 计算决策可信度
        confidence = (new_value - np.mean(self.q_table[state])) / \
                    (np.std(self.q_table[state]) + 1e-6)
        
        # 生成建议文本
        if selected_month == 0:
            timing = "建议立即投资"
        else:
            timing = f"建议在项目开始{selected_month}个月后投资"
            
        return (f"{timing}\n")

def main():
    advisor = InvestmentTimingAdvisor()
    
    # 示例数据
    metrics = MarketMetrics(
        corporate_interest_income=0,
        personal_interest_income=22550,
        non_interest_income=500000,
        investment_return=0.6
    )
    
    # 获取建议
    recommendation = advisor.recommend_timing(metrics)
    print("\n投资时机分析结果:")
    print(recommendation)

if __name__ == "__main__":
    main() 