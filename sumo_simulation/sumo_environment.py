# sumo_simulation/sumo_environment.py
import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import traci
import sumolib
from collections import defaultdict
import logging

class TrafficCorridorSimulation:
    """
    道路走廊模擬環境
    管理多路口交通模擬與資料收集
    """
    
    def __init__(self, sumo_cfg_file="configs/corridor.sumocfg", gui=True):
        self.sumo_cfg = sumo_cfg_file
        self.gui = gui
        self.step_count = 0
        self.intersection_ids = []
        self.traffic_lights = {}
        self.vehicles_data = defaultdict(list)
        self.pedestrian_data = defaultdict(list)
        self.bus_data = defaultdict(list)
        
        # 初始化日誌
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def start_simulation(self):
        """啟動SUMO模擬"""
        sumo_cmd = ["sumo-gui" if self.gui else "sumo", 
                   "-c", self.sumo_cfg,
                   "--waiting-time-memory", "10000",
                   "--time-to-teleport", "-1"]
        
        print(f"執行指令: {' '.join(sumo_cmd)}")
        print(f"當前工作目錄: {os.getcwd()}")
        print(f"設定檔是否存在: {os.path.exists(self.sumo_cfg)}")
        
        try:
            traci.start(sumo_cmd)
            
            # 獲取所有紅綠燈路口ID
            self.intersection_ids = traci.trafficlight.getIDList()
            
            # 初始化紅綠燈狀態記錄
            for tl_id in self.intersection_ids:
                self.traffic_lights[tl_id] = {
                    'current_phase': 0,
                    'phase_duration': 0,
                    'queue_length': 0,
                    'waiting_time': 0
                }
                
            self.logger.info(f"模擬啟動成功，發現 {len(self.intersection_ids)} 個路口")
            
        except Exception as e:
            self.logger.error(f"SUMO啟動失敗: {e}")
            raise
        
    def step(self):
        """執行一個模擬步驟並收集資料"""
        traci.simulationStep()
        self.step_count += 1
        
        # 收集交通資料
        self._collect_traffic_data()
        self._collect_pedestrian_data()
        self._collect_bus_data()
        
        return self._get_system_state()
    
    def _collect_traffic_data(self):
        """收集車輛交通資料"""
        vehicle_ids = traci.vehicle.getIDList()
        
        for vehicle_id in vehicle_ids:
            try:
                pos = traci.vehicle.getPosition(vehicle_id)
                speed = traci.vehicle.getSpeed(vehicle_id)
                waiting_time = traci.vehicle.getWaitingTime(vehicle_id)
                route = traci.vehicle.getRouteID(vehicle_id)
                
                self.vehicles_data[vehicle_id].append({
                    'step': self.step_count,
                    'position': pos,
                    'speed': speed,
                    'waiting_time': waiting_time,
                    'route': route,
                    'timestamp': traci.simulation.getTime()
                })
            except traci.TraCIException:
                continue
                
        # 收集路口資訊
        for tl_id in self.intersection_ids:
            try:
                phase = traci.trafficlight.getPhase(tl_id)
                phase_duration = traci.trafficlight.getPhaseDuration(tl_id)
                
                # 計算各方向排隊長度
                controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                total_queue = sum(traci.lane.getLastStepHaltingNumber(lane) 
                                for lane in controlled_lanes)
                
                self.traffic_lights[tl_id].update({
                    'current_phase': phase,
                    'phase_duration': phase_duration,
                    'queue_length': total_queue,
                    'step': self.step_count
                })
            except traci.TraCIException:
                continue
    
    def _collect_pedestrian_data(self):
        """收集行人資料 (模擬API)"""
        # 這裡模擬行人偵測API的回傳資料
        for tl_id in self.intersection_ids:
            # 模擬行人數量 (0-20人)
            pedestrian_count = np.random.poisson(3)
            # 模擬特殊族群比例 (長者、兒童、團體)
            elderly_ratio = np.random.beta(2, 8)  # 平均約20%
            group_ratio = np.random.beta(1, 9)    # 平均約10%
            
            self.pedestrian_data[tl_id].append({
                'step': self.step_count,
                'pedestrian_count': pedestrian_count,
                'elderly_ratio': elderly_ratio,
                'group_ratio': group_ratio,
                'recommended_green_time': self._calculate_pedestrian_green_time(
                    pedestrian_count, elderly_ratio, group_ratio),
                'timestamp': traci.simulation.getTime()
            })
    
    def _collect_bus_data(self):
        """收集公車資料"""
        vehicle_ids = traci.vehicle.getIDList()
        
        for vehicle_id in vehicle_ids:
            try:
                vehicle_type = traci.vehicle.getVehicleClass(vehicle_id)
                
                if vehicle_type == 'bus':
                    pos = traci.vehicle.getPosition(vehicle_id)
                    speed = traci.vehicle.getSpeed(vehicle_id)
                    route = traci.vehicle.getRouteID(vehicle_id)
                    next_stop = traci.vehicle.getNextStops(vehicle_id)
                    
                    # 預測到站時間
                    eta = self._predict_bus_arrival(vehicle_id, next_stop)
                    
                    self.bus_data[vehicle_id].append({
                        'step': self.step_count,
                        'position': pos,
                        'speed': speed,
                        'route': route,
                        'next_stop': next_stop,
                        'eta': eta,
                        'timestamp': traci.simulation.getTime()
                    })
            except traci.TraCIException:
                continue
    
    def _calculate_pedestrian_green_time(self, count, elderly_ratio, group_ratio):
        """計算適應性行人綠燈時間 (15-60秒)"""
        base_time = 20  # 基礎通行時間
        
        # 人數影響 (每人增加1秒)
        count_factor = min(count * 1, 25)
        
        # 長者影響 (增加50%時間)
        elderly_factor = elderly_ratio * count * 0.5
        
        # 團體影響 (團體需要更多時間聚集)
        group_factor = group_ratio * count * 0.3
        
        total_time = base_time + count_factor + elderly_factor + group_factor
        return max(15, min(60, int(total_time)))
    
    def _predict_bus_arrival(self, bus_id, stops):
        """預測公車到站時間"""
        if not stops:
            return None
            
        try:
            bus_pos = traci.vehicle.getPosition(bus_id)
            bus_speed = max(traci.vehicle.getSpeed(bus_id), 5)  # 最低5m/s
            
            stop_pos = traci.busStop.getStartPos(stops[0].stoppingPlaceID)
            distance = abs(stop_pos - traci.vehicle.getLanePosition(bus_id))
            
            eta = distance / bus_speed
            return eta
        except:
            return None
    
    def _get_system_state(self):
        """獲取系統當前狀態"""
        state = {
            'step': self.step_count,
            'timestamp': traci.simulation.getTime(),
            'traffic_lights': dict(self.traffic_lights),
            'vehicle_count': len(traci.vehicle.getIDList()),
            'active_intersections': len(self.intersection_ids)
        }
        
        # 計算系統級指標
        total_waiting_vehicles = sum(
            tl['queue_length'] for tl in self.traffic_lights.values()
        )
        
        avg_speed = 0
        vehicle_count = len(traci.vehicle.getIDList())
        if vehicle_count > 0:
            try:
                speeds = [traci.vehicle.getSpeed(vid) for vid in traci.vehicle.getIDList()]
                avg_speed = np.mean(speeds)
            except:
                avg_speed = 0
        
        state.update({
            'total_waiting_vehicles': total_waiting_vehicles,
            'average_speed': avg_speed,
            'system_efficiency': self._calculate_efficiency()
        })
        
        return state
    
    def _calculate_efficiency(self):
        """計算系統效率指標"""
        try:
            # 簡化效率計算：基於平均速度和等待時間
            total_vehicles = len(traci.vehicle.getIDList())
            if total_vehicles == 0:
                return 1.0
                
            total_waiting = sum(
                traci.vehicle.getWaitingTime(vid) 
                for vid in traci.vehicle.getIDList()
            )
            
            avg_waiting = total_waiting / total_vehicles if total_vehicles > 0 else 0
            
            # 效率 = 1 - (平均等待時間 / 最大可接受等待時間)
            max_acceptable_wait = 120  # 2分鐘
            efficiency = max(0, 1 - (avg_waiting / max_acceptable_wait))
            
            return efficiency
        except:
            return 0.5  # 預設值
    
    def set_traffic_light_phase(self, tl_id, phase_index):
        """設置紅綠燈相位"""
        try:
            traci.trafficlight.setPhase(tl_id, phase_index)
        except traci.TraCIException as e:
            self.logger.warning(f"設置路口 {tl_id} 相位失敗: {e}")
    
    def get_green_wave_recommendations(self):
        """獲取綠波協調建議"""
        recommendations = {}
        
        for tl_id in self.intersection_ids:
            # 獲取路口資料
            tl_data = self.traffic_lights[tl_id]
            
            # 簡化綠波計算 (實際應使用更複雜的演算法)
            queue_density = tl_data['queue_length']
            
            if queue_density > 5:  # 高負荷
                recommended_phase = 2  # 主要方向綠燈
                duration = 45
            elif queue_density > 2:  # 中等負荷
                recommended_phase = 1  # 平衡相位
                duration = 35
            else:  # 低負荷
                recommended_phase = 0  # 標準相位
                duration = 25
            
            recommendations[tl_id] = {
                'phase': recommended_phase,
                'duration': duration,
                'confidence': min(1.0, queue_density / 10)
            }
        
        return recommendations
    
    def close(self):
        """關閉模擬"""
        try:
            traci.close()
            self.logger.info("模擬已關閉")
        except:
            pass
        
    def export_data(self, filename):
        """匯出收集的資料"""
        import pickle
        
        data = {
            'vehicles': dict(self.vehicles_data),
            'pedestrians': dict(self.pedestrian_data),
            'buses': dict(self.bus_data),
            'traffic_lights': dict(self.traffic_lights),
            'total_steps': self.step_count
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
            
        self.logger.info(f"資料已匯出至 {filename}")

# 使用範例
if __name__ == "__main__":
    # 需要先準備SUMO設定檔
    sim = TrafficCorridorSimulation("configs/corridor.sumocfg", gui=True)
    
    try:
        sim.start_simulation()
        
        # 執行100步模擬
        for i in range(100):
            state = sim.step()
            
            if i % 20 == 0:
                print(f"Step {i}: 效率={state['system_efficiency']:.3f}, "
                      f"平均速度={state['average_speed']:.2f}m/s, "
                      f"車輛數={state['vehicle_count']}")
                
                # 每20步獲取綠波建議
                recommendations = sim.get_green_wave_recommendations()
                print(f"綠波建議: {len(recommendations)} 個路口")
        
        # 匯出資料
        sim.export_data("simulation_data.pkl")
        print("模擬完成")
        
    except Exception as e:
        print(f"模擬過程發生錯誤: {e}")
    finally:
        sim.close()