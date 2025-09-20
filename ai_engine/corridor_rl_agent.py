# ai_engine/corridor_rl_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from typing import Dict, List, Tuple, Optional
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiIntersectionDQN(nn.Module):
    """
    多路口協調的深度Q網路
    輸入：所有路口的交通狀態
    輸出：所有路口的最佳相位動作
    """
    
    def __init__(self, state_dim: int, action_dim: int, n_intersections: int):
        super(MultiIntersectionDQN, self).__init__()
        self.n_intersections = n_intersections
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 共享特徵提取層
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 每個路口的專用輸出層
        self.intersection_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, action_dim)
            ) for _ in range(n_intersections)
        ])
        
        # 協調層 (考慮路口間相互影響)
        self.coordination_layer = nn.Sequential(
            nn.Linear(64 * n_intersections, 128),
            nn.ReLU(),
            nn.Linear(128, n_intersections * action_dim)
        )
        
    def forward(self, x):
        # x shape: (batch_size, state_dim)
        shared_features = self.shared_layers(x)
        
        # 單獨預測每個路口
        individual_q_values = []
        for i, head in enumerate(self.intersection_heads):
            q_vals = head(shared_features)
            individual_q_values.append(q_vals)
        
        # 協調預測
        coord_input = shared_features.repeat(1, self.n_intersections)
        coord_q_values = self.coordination_layer(coord_input)
        coord_q_values = coord_q_values.view(-1, self.n_intersections, self.action_dim)
        
        # 結合個別和協調預測
        final_q_values = []
        for i in range(self.n_intersections):
            combined = individual_q_values[i] + coord_q_values[:, i, :]
            final_q_values.append(combined)
        
        return torch.stack(final_q_values, dim=1)  # (batch_size, n_intersections, action_dim)

class CorridorRLAgent:
    """
    道路走廊強化學習智能體
    負責多路口協調控制
    """
    
    def __init__(self, 
                 n_intersections: int = 5,
                 state_features_per_intersection: int = 12,
                 n_actions_per_intersection: int = 4,
                 learning_rate: float = 0.001,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 memory_size: int = 10000,
                 batch_size: int = 32,
                 target_update_freq: int = 100):
        
        self.n_intersections = n_intersections
        self.state_dim = state_features_per_intersection * n_intersections
        self.action_dim = n_actions_per_intersection
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # 設備選擇
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用設備: {self.device}")
        
        # 建立網路
        self.q_network = MultiIntersectionDQN(
            self.state_dim, self.action_dim, n_intersections
        ).to(self.device)
        
        self.target_network = MultiIntersectionDQN(
            self.state_dim, self.action_dim, n_intersections
        ).to(self.device)
        
        # 初始化目標網路
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 優化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 經驗回放緩衝區
        Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])
        self.memory = deque(maxlen=memory_size)
        self.Experience = Experience
        
        # 訓練統計
        self.training_step = 0
        self.episode_rewards = []
        self.losses = []
        
        # 綠波參數
        self.green_wave_params = {
            'speed_limit': 16.67,  # m/s (60 km/h)
            'intersection_distance': 500,  # m
            'coordination_window': 3  # 考慮前後3個路口
        }
        
    def get_state_features(self, intersection_data: Dict, system_state: Dict) -> np.ndarray:
        """
        從模擬數據中提取狀態特徵
        """
        features = []
        
        for i in range(self.n_intersections):
            tl_id = f"J{i+1}"
            
            if tl_id in intersection_data:
                tl_data = intersection_data[tl_id]
                
                # 基礎交通特徵
                queue_length = tl_data.get('queue_length', 0)
                current_phase = tl_data.get('current_phase', 0)
                phase_duration = tl_data.get('phase_duration', 0)
                waiting_time = tl_data.get('waiting_time', 0)
                
                # 標準化
                queue_length_norm = min(queue_length / 20.0, 1.0)  # 假設最大20輛車
                phase_norm = current_phase / 3.0  # 假設4個相位 (0-3)
                duration_norm = min(phase_duration / 120.0, 1.0)  # 最大120秒
                waiting_norm = min(waiting_time / 60.0, 1.0)  # 最大60秒
                
                # 時段特徵 (模擬)
                time_of_day = (system_state.get('timestamp', 0) % 86400) / 86400.0
                is_peak_hour = 1.0 if (7*3600 <= system_state.get('timestamp', 0) % 86400 <= 9*3600) or \
                                     (17*3600 <= system_state.get('timestamp', 0) % 86400 <= 19*3600) else 0.0
                
                # 上下游狀況
                upstream_queue = 0
                downstream_queue = 0
                
                if i > 0:  # 有上游路口
                    upstream_id = f"J{i}"
                    if upstream_id in intersection_data:
                        upstream_queue = min(intersection_data[upstream_id].get('queue_length', 0) / 20.0, 1.0)
                
                if i < self.n_intersections - 1:  # 有下游路口
                    downstream_id = f"J{i+2}"
                    if downstream_id in intersection_data:
                        downstream_queue = min(intersection_data[downstream_id].get('queue_length', 0) / 20.0, 1.0)
                
                # 行人需求 (從模擬系統獲取)
                pedestrian_demand = 0.3  # 預設值
                
                # 公車因子
                bus_priority = 0.0  # 預設值
                
                # 緊急車輛
                emergency_factor = 0.0  # 預設值
                
                intersection_features = [
                    queue_length_norm,
                    phase_norm,
                    duration_norm,
                    waiting_norm,
                    time_of_day,
                    is_peak_hour,
                    upstream_queue,
                    downstream_queue,
                    pedestrian_demand,
                    bus_priority,
                    emergency_factor,
                    float(i) / self.n_intersections  # 位置編碼
                ]
            else:
                # 如果沒有數據，使用預設值
                intersection_features = [0.0] * 12
            
            features.extend(intersection_features)
        
        return np.array(features, dtype=np.float32)
    
    def act(self, state: np.ndarray, training: bool = True) -> List[int]:
        """
        根據狀態選擇動作 (epsilon-greedy策略)
        返回每個路口的相位選擇
        """
        if training and random.random() < self.epsilon:
            # 隨機探索
            actions = [random.randint(0, self.action_dim - 1) for _ in range(self.n_intersections)]
            return actions
        
        # 貪婪選擇
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)  # (1, n_intersections, action_dim)
            actions = q_values.argmax(dim=2).cpu().numpy()[0]  # (n_intersections,)
        
        return actions.tolist()
    
    def remember(self, state, action, reward, next_state, done):
        """儲存經驗到回放緩衝區"""
        self.memory.append(self.Experience(state, action, reward, next_state, done))
    
    def replay(self) -> Optional[float]:
        """從經驗回放中學習"""
        if len(self.memory) < self.batch_size:
            return None
        
        # 隨機抽樣
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
        
        # 當前Q值
        current_q_values = self.q_network(states)
        # 使用 gather 選擇對應動作的Q值
        current_q_values = current_q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        
        # 目標Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_values = next_q_values.max(dim=2)[0]
            target_q_values = rewards + (0.99 * max_next_q_values * ~dones)
        
        # 計算損失
        loss = nn.MSELoss()(current_q_values.sum(dim=1), target_q_values.sum(dim=1))
        
        # 反向傳播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # 更新epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 定期更新目標網路
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.losses.append(loss.item())
        return loss.item()
    
    def calculate_reward(self, 
                        prev_state: Dict, 
                        current_state: Dict, 
                        actions: List[int],
                        pedestrian_data: Dict,
                        bus_data: Dict) -> float:
        """
        計算綜合獎勵函數
        考慮：通行效率、行人服務、公車優先、綠波協調
        """
        reward = 0.0
        
        # 1. 通行效率獎勵
        efficiency_reward = self._calculate_efficiency_reward(prev_state, current_state)
        
        # 2. 行人服務獎勵  
        pedestrian_reward = self._calculate_pedestrian_reward(pedestrian_data)
        
        # 3. 公車優先獎勵
        bus_reward = self._calculate_bus_priority_reward(bus_data)
        
        # 4. 綠波協調獎勵
        coordination_reward = self._calculate_coordination_reward(actions, current_state)
        
        # 5. 安全性獎勵 (避免急停急起)
        safety_reward = self._calculate_safety_reward(prev_state, current_state)
        
        # 加權組合
        total_reward = (
            0.4 * efficiency_reward +
            0.2 * pedestrian_reward +
            0.2 * bus_reward +
            0.15 * coordination_reward +
            0.05 * safety_reward
        )
        
        return total_reward
    
    def _calculate_efficiency_reward(self, prev_state: Dict, current_state: Dict) -> float:
        """計算通行效率獎勵"""
        prev_efficiency = prev_state.get('system_efficiency', 0)
        current_efficiency = current_state.get('system_efficiency', 0)
        
        efficiency_improvement = current_efficiency - prev_efficiency
        
        # 額外獎勵低等待時間
        total_waiting = sum(
            tl.get('queue_length', 0) 
            for tl in current_state.get('traffic_lights', {}).values()
        )
        
        waiting_penalty = -total_waiting * 0.1
        
        return efficiency_improvement * 10 + waiting_penalty
    
    def _calculate_pedestrian_reward(self, pedestrian_data: Dict) -> float:
        """計算行人服務獎勵"""
        reward = 0.0
        
        for intersection_id, ped_data in pedestrian_data.items():
            if not ped_data:
                continue
                
            latest_data = ped_data[-1]
            pedestrian_count = latest_data.get('pedestrian_count', 0)
            recommended_time = latest_data.get('recommended_green_time', 30)
            
            # 獎勵適當的行人服務
            if pedestrian_count > 0:
                # 基礎服務獎勵
                service_reward = min(pedestrian_count * 0.5, 5.0)
                
                # 特殊族群額外獎勵
                elderly_ratio = latest_data.get('elderly_ratio', 0)
                if elderly_ratio > 0.3:  # 長者比例高
                    service_reward += 2.0
                
                reward += service_reward
        
        return reward
    
    def _calculate_bus_priority_reward(self, bus_data: Dict) -> float:
        """計算公車優先獎勵"""
        reward = 0.0
        
        for bus_id, bus_history in bus_data.items():
            if not bus_history:
                continue
                
            latest_data = bus_history[-1]
            eta = latest_data.get('eta', None)
            
            if eta is not None and eta < 60:  # 1分鐘內到站
                # 公車接近時給予獎勵
                proximity_reward = max(0, (60 - eta) / 60 * 3.0)
                reward += proximity_reward
        
        return reward
    
    def _calculate_coordination_reward(self, actions: List[int], current_state: Dict) -> float:
        """計算綠波協調獎勵"""
        reward = 0.0
        
        # 檢查相鄰路口的相位協調
        for i in range(len(actions) - 1):
            current_phase = actions[i]
            next_phase = actions[i + 1]
            
            # 如果相鄰路口採用協調的相位，給予獎勵
            if self._is_coordinated_phase(current_phase, next_phase, i):
                reward += 2.0
        
        # 檢查整體綠波品質
        wave_quality = self._evaluate_green_wave_quality(actions)
        reward += wave_quality * 5.0
        
        return reward
    
    def _calculate_safety_reward(self, prev_state: Dict, current_state: Dict) -> float:
        """計算安全性獎勵"""
        reward = 0.0
        
        # 獎勵平穩的相位變化
        prev_avg_speed = prev_state.get('average_speed', 0)
        current_avg_speed = current_state.get('average_speed', 0)
        
        speed_change = abs(current_avg_speed - prev_avg_speed)
        
        if speed_change < 2.0:  # 速度變化小於2m/s
            reward += 1.0
        else:
            reward -= speed_change * 0.5  # 懲罰劇烈變化
        
        return reward
    
    def _is_coordinated_phase(self, phase1: int, phase2: int, position: int) -> bool:
        """判斷兩個相位是否協調"""
        # 簡化的協調判斷邏輯
        # 實際應考慮距離、速度、相位差等
        
        distance = self.green_wave_params['intersection_distance']
        speed = self.green_wave_params['speed_limit']
        travel_time = distance / speed
        
        # 基本綠波協調：主要方向同步
        if phase1 == phase2 and phase1 in [0, 2]:  # 主要方向相位
            return True
        
        # 考慮行進時間的相位差
        optimal_phase_diff = int(travel_time / 30)  # 假設每相位30秒
        actual_phase_diff = abs(phase1 - phase2)
        
        return actual_phase_diff <= optimal_phase_diff
    
    def _evaluate_green_wave_quality(self, actions: List[int]) -> float:
        """評估綠波品質"""
        if len(actions) < 2:
            return 0.0
        
        # 計算相位一致性
        main_direction_count = sum(1 for action in actions if action in [0, 2])
        consistency_ratio = main_direction_count / len(actions)
        
        # 計算相位變化的平滑度
        phase_changes = sum(1 for i in range(len(actions)-1) 
                           if abs(actions[i] - actions[i+1]) > 2)
        smoothness = 1.0 - (phase_changes / max(1, len(actions)-1))
        
        return (consistency_ratio + smoothness) / 2.0
    
    def get_green_wave_timing(self, intersection_positions: List[float], 
                            target_speed: float) -> List[float]:
        """
        計算綠波時序
        """
        if not intersection_positions:
            return []
        
        # 以第一個路口為基準
        base_position = intersection_positions[0]
        timings = []
        
        for position in intersection_positions:
            distance = abs(position - base_position)
            travel_time = distance / target_speed
            timings.append(travel_time)
        
        return timings
    
    def save_model(self, filepath: str):
        """儲存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode_rewards': self.episode_rewards,
            'losses': self.losses
        }, filepath)
        logger.info(f"模型已儲存至 {filepath}")
    
    def load_model(self, filepath: str):
        """載入模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.losses = checkpoint.get('losses', [])
        
        logger.info(f"模型已從 {filepath} 載入")

class CoordinatorManager:
    """
    協調器管理員
    整合各功能模組的決策
    """
    
    def __init__(self, rl_agent: CorridorRLAgent):
        self.rl_agent = rl_agent
        self.pedestrian_coordinator = PedestrianCoordinator()
        self.bus_coordinator = BusCoordinator()
        self.emergency_coordinator = EmergencyCoordinator()
        
        self.logger = logging.getLogger(__name__)
    
    def make_decision(self, 
                     system_state: Dict,
                     pedestrian_data: Dict,
                     bus_data: Dict,
                     emergency_data: Dict) -> Dict:
        """
        綜合決策
        優先級：緊急車輛 > 公車優先 > 行人需求 > 一般交通
        """
        
        # 1. 檢查緊急情況
        emergency_actions = self.emergency_coordinator.get_emergency_actions(emergency_data)
        if emergency_actions:
            return {
                'actions': emergency_actions,
                'type': 'emergency',
                'reason': '緊急車輛通行'
            }
        
        # 2. 獲取RL基礎建議
        state_features = self.rl_agent.get_state_features(
            system_state.get('traffic_lights', {}), 
            system_state
        )
        base_actions = self.rl_agent.act(state_features, training=False)
        
        # 3. 行人需求調整
        pedestrian_adjustments = self.pedestrian_coordinator.get_adjustments(
            pedestrian_data, base_actions
        )
        
        # 4. 公車優先調整
        bus_adjustments = self.bus_coordinator.get_adjustments(
            bus_data, base_actions, pedestrian_adjustments
        )
        
        # 5. 整合最終決策
        final_actions = self._integrate_adjustments(
            base_actions, pedestrian_adjustments, bus_adjustments
        )
        
        return {
            'actions': final_actions,
            'type': 'coordinated',
            'base_actions': base_actions,
            'pedestrian_adjustments': pedestrian_adjustments,
            'bus_adjustments': bus_adjustments
        }
    
    def _integrate_adjustments(self, 
                              base_actions: List[int],
                              pedestrian_adj: Dict,
                              bus_adj: Dict) -> List[int]:
        """整合各種調整"""
        final_actions = base_actions.copy()
        
        # 應用行人調整
        for intersection_idx, adjustment in pedestrian_adj.items():
            if adjustment.get('extend_green', False):
                # 延長綠燈時間 (這裡簡化為相位選擇)
                final_actions[intersection_idx] = adjustment['recommended_phase']
        
        # 應用公車調整 (優先級較高)
        for intersection_idx, adjustment in bus_adj.items():
            if adjustment.get('priority', False):
                final_actions[intersection_idx] = adjustment['recommended_phase']
        
        return final_actions

class PedestrianCoordinator:
    """行人需求協調器"""
    
    def get_adjustments(self, pedestrian_data: Dict, base_actions: List[int]) -> Dict:
        """根據行人資料提供調整建議"""
        adjustments = {}
        
        for i, (intersection_id, ped_data) in enumerate(pedestrian_data.items()):
            if not ped_data:
                continue
                
            latest_data = ped_data[-1]
            pedestrian_count = latest_data.get('pedestrian_count', 0)
            elderly_ratio = latest_data.get('elderly_ratio', 0)
            recommended_time = latest_data.get('recommended_green_time', 30)
            
            # 判斷是否需要調整
            if pedestrian_count > 10 or elderly_ratio > 0.4:
                adjustments[i] = {
                    'extend_green': True,
                    'recommended_phase': 1,  # 行人相位
                    'duration': recommended_time,
                    'reason': f'行人數量: {pedestrian_count}, 長者比例: {elderly_ratio:.2f}'
                }
        
        return adjustments

class BusCoordinator:
    """公車協調器"""
    
    def get_adjustments(self, bus_data: Dict, base_actions: List[int], 
                       pedestrian_adj: Dict) -> Dict:
        """根據公車資料提供調整建議"""
        adjustments = {}
        
        for bus_id, bus_history in bus_data.items():
            if not bus_history:
                continue
                
            latest_data = bus_history[-1]
            eta = latest_data.get('eta', None)
            
            if eta is not None and eta < 30:  # 30秒內到站
                # 確定影響的路口
                affected_intersection = self._get_affected_intersection(latest_data)
                
                if affected_intersection is not None:
                    # 檢查與行人需求的衝突
                    if affected_intersection not in pedestrian_adj:
                        adjustments[affected_intersection] = {
                            'priority': True,
                            'recommended_phase': 0,  # 主要方向相位
                            'reason': f'公車 {bus_id} 即將到站 (ETA: {eta:.1f}s)'
                        }
        
        return adjustments
    
    def _get_affected_intersection(self, bus_data: Dict) -> Optional[int]:
        """根據公車位置確定影響的路口"""
        # 簡化實現：根據路線和位置推斷
        route = bus_data.get('route', '')
        position = bus_data.get('position', (0, 0))
        
        # 根據X座標判斷最近的路口
        x_coord = position[0]
        intersection_positions = [0, 500, 1000, 1500, 2000]
        
        distances = [abs(x_coord - pos) for pos in intersection_positions]
        closest_intersection = distances.index(min(distances))
        
        return closest_intersection

class EmergencyCoordinator:
    """緊急車輛協調器"""
    
    def get_emergency_actions(self, emergency_data: Dict) -> Optional[List[int]]:
        """處理緊急車輛通行"""
        if not emergency_data or not emergency_data.get('active', False):
            return None
        
        emergency_route = emergency_data.get('route', [])
        if not emergency_route:
            return None
        
        # 為緊急車輛清道：所有路口設為主要方向綠燈
        emergency_actions = [0] * 5  # 假設5個路口，都設為相位0
        
        return emergency_actions

# 使用範例和測試
if __name__ == "__main__":
    # 建立RL智能體
    agent = CorridorRLAgent(
        n_intersections=5,
        learning_rate=0.001,
        epsilon=0.1
    )
    
    # 建立協調器
    coordinator = CoordinatorManager(agent)
    
    # 模擬狀態資料
    system_state = {
        'step': 1000,
        'timestamp': 28800,  # 8:00 AM
        'traffic_lights': {
            'J1': {'queue_length': 8, 'current_phase': 0, 'phase_duration': 25, 'waiting_time': 15},
            'J2': {'queue_length': 12, 'current_phase': 1, 'phase_duration': 35, 'waiting_time': 22},
            'J3': {'queue_length': 6, 'current_phase': 0, 'phase_duration': 20, 'waiting_time': 8},
            'J4': {'queue_length': 15, 'current_phase': 2, 'phase_duration': 40, 'waiting_time': 35},
            'J5': {'queue_length': 4, 'current_phase': 0, 'phase_duration': 15, 'waiting_time': 5}
        },
        'average_speed': 12.5,
        'system_efficiency': 0.75
    }
    
    pedestrian_data = {
        'J2': [{'pedestrian_count': 15, 'elderly_ratio': 0.3, 'recommended_green_time': 45}],
        'J4': [{'pedestrian_count': 8, 'elderly_ratio': 0.1, 'recommended_green_time': 30}]
    }
    
    bus_data = {
        'bus_001': [{'eta': 25, 'route': 'bus_route_1', 'position': (450, 0)}]
    }
    
    emergency_data = {'active': False}
    
    # 進行決策
    decision = coordinator.make_decision(
        system_state, pedestrian_data, bus_data, emergency_data
    )
    
    print("智慧協調決策結果:")
    print(f"決策類型: {decision['type']}")
    print(f"最終動作: {decision['actions']}")
    
    if 'base_actions' in decision:
        print(f"RL基礎建議: {decision['base_actions']}")
        print(f"行人調整: {decision['pedestrian_adjustments']}")
        print(f"公車調整: {decision['bus_adjustments']}")
    
    # 模擬訓練過程
    print("\n開始訓練...")
    for episode in range(10):
        state_features = agent.get_state_features(
            system_state['traffic_lights'], 
            system_state
        )
        
        actions = agent.act(state_features, training=True)
        
        # 模擬獎勵計算
        reward = agent.calculate_reward(
            system_state, system_state, actions, 
            pedestrian_data, bus_data
        )
        
        print(f"Episode {episode}: Actions={actions}, Reward={reward:.3f}, Epsilon={agent.epsilon:.3f}")
        
        # 模擬經驗儲存和學習
        next_state = state_features  # 簡化
        agent.remember(state_features, actions, reward, next_state, False)
        
        if len(agent.memory) >= agent.batch_size:
            loss = agent.replay()
            if loss:
                print(f"  訓練損失: {loss:.4f}")
    
    print("訓練完成！")