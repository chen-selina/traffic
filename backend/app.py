# backend/app.py
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import json
import threading
import time
import numpy as np
from datetime import datetime
import sqlite3
import logging
from typing import Dict, List, Optional
import paho.mqtt.client as mqtt
import queue
import uuid

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'traffic-corridor-secret-key'
CORS(app, origins="*")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

class TrafficDataManager:
    """交通資料管理器"""
    
    def __init__(self):
        self.db_path = 'data/traffic_corridor.db'
        
        # 確保data目錄存在
        import os
        os.makedirs('data', exist_ok=True)
        
        self.init_database()
        self.current_data = {
            'system_state': {},
            'traffic_lights': {},
            'pedestrian_data': {},
            'bus_data': {},
            'emergency_data': {'active': False},
            'kpi_metrics': {},
            'last_update': None
        }
        
        # 資料流隊列
        self.data_queue = queue.Queue(maxsize=1000)
        
        # 啟動背景處理線程
        self.processing_thread = threading.Thread(target=self._background_processor, daemon=True)
        self.processing_thread.start()
        
    def init_database(self):
        """初始化資料庫"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 系統狀態表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                step INTEGER,
                vehicle_count INTEGER,
                average_speed REAL,
                system_efficiency REAL,
                total_waiting_vehicles INTEGER,
                data_json TEXT
            )
        ''')
        
        # 路口狀態表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intersection_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                intersection_id TEXT,
                current_phase INTEGER,
                phase_duration REAL,
                queue_length INTEGER,
                waiting_time REAL,
                throughput INTEGER
            )
        ''')
        
        # KPI指標表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS kpi_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                metric_name TEXT,
                metric_value REAL,
                intersection_id TEXT
            )
        ''')
        
        # 事件記錄表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                event_type TEXT,
                event_data TEXT,
                intersection_id TEXT,
                severity TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("資料庫初始化完成")
    
    def _background_processor(self):
        """背景資料處理線程"""
        while True:
            try:
                if not self.data_queue.empty():
                    data_item = self.data_queue.get(timeout=1)
                    self._process_data_item(data_item)
                    
                    # 發送即時更新到前端
                    socketio.emit('data_update', self.current_data)
                    
                time.sleep(0.1)  # 避免過度消耗CPU
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"背景處理錯誤: {e}")
    
    def _process_data_item(self, data_item):
        """處理單個資料項目"""
        data_type = data_item.get('type')
        
        if data_type == 'system_state':
            self._update_system_state(data_item['data'])
        elif data_type == 'intersection_update':
            self._update_intersection_data(data_item['data'])
        elif data_type == 'pedestrian_update':
            self._update_pedestrian_data(data_item['data'])
        elif data_type == 'bus_update':
            self._update_bus_data(data_item['data'])
        elif data_type == 'emergency_event':
            self._handle_emergency_event(data_item['data'])
        
        self.current_data['last_update'] = datetime.now().isoformat()
    
    def _update_system_state(self, state_data):
        """更新系統狀態"""
        self.current_data['system_state'] = state_data
        
        # 儲存到資料庫
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO system_states 
            (timestamp, step, vehicle_count, average_speed, system_efficiency, 
             total_waiting_vehicles, data_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            time.time(),
            state_data.get('step', 0),
            state_data.get('vehicle_count', 0),
            state_data.get('average_speed', 0),
            state_data.get('system_efficiency', 0),
            state_data.get('total_waiting_vehicles', 0),
            json.dumps(state_data)
        ))
        
        conn.commit()
        conn.close()
        
        # 計算KPI
        self._calculate_kpi_metrics(state_data)
    
    def _update_intersection_data(self, intersection_data):
        """更新路口資料"""
        for intersection_id, data in intersection_data.items():
            self.current_data['traffic_lights'][intersection_id] = data
            
            # 儲存到資料庫
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO intersection_states 
                (timestamp, intersection_id, current_phase, phase_duration, 
                 queue_length, waiting_time, throughput)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                time.time(),
                intersection_id,
                data.get('current_phase', 0),
                data.get('phase_duration', 0),
                data.get('queue_length', 0),
                data.get('waiting_time', 0),
                data.get('throughput', 0)
            ))
            
            conn.commit()
            conn.close()
    
    def _update_pedestrian_data(self, pedestrian_data):
        """更新行人資料"""
        self.current_data['pedestrian_data'] = pedestrian_data
    
    def _update_bus_data(self, bus_data):
        """更新公車資料"""
        self.current_data['bus_data'] = bus_data
    
    def _handle_emergency_event(self, emergency_data):
        """處理緊急事件"""
        self.current_data['emergency_data'] = emergency_data
        
        # 記錄事件
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO events (timestamp, event_type, event_data, severity)
            VALUES (?, ?, ?, ?)
        ''', (
            time.time(),
            'emergency',
            json.dumps(emergency_data),
            'high'
        ))
        
        conn.commit()
        conn.close()
        
        # 立即發送緊急通知
        socketio.emit('emergency_alert', emergency_data)
    
    def _calculate_kpi_metrics(self, state_data):
        """計算KPI指標"""
        current_time = time.time()
        
        # 系統級KPI
        metrics = {
            'average_speed': state_data.get('average_speed', 0),
            'system_efficiency': state_data.get('system_efficiency', 0),
            'total_delay': self._calculate_total_delay(),
            'throughput': self._calculate_throughput(),
            'green_wave_effectiveness': self._calculate_green_wave_effectiveness()
        }
        
        self.current_data['kpi_metrics'] = metrics
        
        # 儲存到資料庫
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for metric_name, metric_value in metrics.items():
            cursor.execute('''
                INSERT INTO kpi_metrics (timestamp, metric_name, metric_value)
                VALUES (?, ?, ?)
            ''', (current_time, metric_name, metric_value))
        
        conn.commit()
        conn.close()
    
    def _calculate_total_delay(self):
        """計算總延遲時間"""
        total_delay = sum(
            tl_data.get('waiting_time', 0) * tl_data.get('queue_length', 0)
            for tl_data in self.current_data['traffic_lights'].values()
        )
        return total_delay
    
    def _calculate_throughput(self):
        """計算系統通行量"""
        # 簡化計算：基於車輛數和平均速度
        vehicle_count = self.current_data['system_state'].get('vehicle_count', 0)
        avg_speed = self.current_data['system_state'].get('average_speed', 0)
        return vehicle_count * avg_speed / 100  # 標準化
    
    def _calculate_green_wave_effectiveness(self):
        """計算綠波效果"""
        # 簡化計算：基於相位一致性
        phases = [
            tl_data.get('current_phase', 0) 
            for tl_data in self.current_data['traffic_lights'].values()
        ]
        
        if not phases:
            return 0
        
        # 計算主要方向相位的比例
        main_direction_phases = sum(1 for phase in phases if phase in [0, 2])
        return main_direction_phases / len(phases)
    
    def add_data(self, data_type, data):
        """添加資料到處理隊列"""
        try:
            self.data_queue.put({
                'type': data_type,
                'data': data,
                'timestamp': time.time()
            }, timeout=1)
        except queue.Full:
            logger.warning("資料隊列已滿，丟棄舊資料")
    
    def get_historical_data(self, hours=24):
        """獲取歷史資料"""
        end_time = time.time()
        start_time = end_time - (hours * 3600)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 系統狀態歷史
        cursor.execute('''
            SELECT timestamp, average_speed, system_efficiency, total_waiting_vehicles
            FROM system_states 
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        ''', (start_time, end_time))
        
        system_history = cursor.fetchall()
        
        # KPI指標歷史
        cursor.execute('''
            SELECT timestamp, metric_name, metric_value
            FROM kpi_metrics 
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        ''', (start_time, end_time))
        
        kpi_history = cursor.fetchall()
        
        conn.close()
        
        return {
            'system_history': system_history,
            'kpi_history': kpi_history
        }

# 初始化資料管理器
data_manager = TrafficDataManager()

# Fake API 服務類別
class FakeAPIServices:
    """模擬外部API服務"""
    
    def __init__(self):
        self.pedestrian_detector = PedestrianDetectorAPI()
        self.emergency_detector = EmergencyVehicleDetectorAPI()
        self.weather_service = WeatherServiceAPI()
        
    def get_all_fake_data(self):
        """獲取所有模擬資料"""
        return {
            'pedestrian': self.pedestrian_detector.get_detection_data(),
            'emergency': self.emergency_detector.get_detection_data(),
            'weather': self.weather_service.get_weather_data()
        }

class PedestrianDetectorAPI:
    """模擬行人偵測API"""
    
    def get_detection_data(self):
        """模擬行人偵測資料"""
        intersections = ['J1', 'J2', 'J3', 'J4', 'J5']
        pedestrian_data = {}
        
        for intersection in intersections:
            # 模擬行人數量 (時段影響)
            hour = datetime.now().hour
            base_count = 5 if 7 <= hour <= 9 or 17 <= hour <= 19 else 2
            pedestrian_count = max(0, int(np.random.poisson(base_count)))
            
            # 模擬族群特徵
            elderly_ratio = max(0, min(1, np.random.beta(2, 5)))
            child_ratio = max(0, min(1, np.random.beta(1, 9)))
            group_size = max(1, int(np.random.gamma(1.5, 2)))
            
            # 計算推薦綠燈時間
            base_time = 20
            count_factor = pedestrian_count * 1.5
            elderly_factor = elderly_ratio * pedestrian_count * 2
            group_factor = (group_size - 1) * 3
            
            recommended_time = min(60, max(15, int(base_time + count_factor + elderly_factor + group_factor)))
            
            pedestrian_data[intersection] = [{
                'timestamp': time.time(),
                'pedestrian_count': pedestrian_count,
                'elderly_ratio': elderly_ratio,
                'child_ratio': child_ratio,
                'group_size': group_size,
                'recommended_green_time': recommended_time,
                'confidence': np.random.uniform(0.8, 0.98)
            }]
        
        return pedestrian_data

class EmergencyVehicleDetectorAPI:
    """模擬緊急車輛偵測API"""
    
    def __init__(self):
        self.emergency_active = False
        self.emergency_start_time = None
        self.emergency_duration = 0
        
    def get_detection_data(self):
        """模擬緊急車輛偵測"""
        current_time = time.time()
        
        # 隨機觸發緊急事件 (低機率)
        if not self.emergency_active and np.random.random() < 0.001:  # 0.1% 機率
            self.emergency_active = True
            self.emergency_start_time = current_time
            self.emergency_duration = np.random.uniform(60, 180)  # 1-3分鐘
            logger.info("模擬緊急車輛事件觸發")
        
        # 檢查緊急事件是否結束
        if self.emergency_active and (current_time - self.emergency_start_time) > self.emergency_duration:
            self.emergency_active = False
            logger.info("模擬緊急車輛事件結束")
        
        if self.emergency_active:
            # 模擬緊急車輛位置和路線
            elapsed_time = current_time - self.emergency_start_time
            progress = elapsed_time / self.emergency_duration
            
            # 假設沿主要走廊移動
            position_x = progress * 2000  # 從0移動到2000m
            
            return {
                'active': True,
                'vehicle_type': 'ambulance',
                'position': (position_x, 0),
                'route': ['J1', 'J2', 'J3', 'J4', 'J5'],
                'eta_intersections': self._calculate_eta(position_x),
                'priority_level': 'high',
                'confidence': 0.95
            }
        else:
            return {'active': False}
    
    def _calculate_eta(self, current_position):
        """計算到各路口的ETA"""
        intersection_positions = [0, 500, 1000, 1500, 2000]
        speed = 20  # 緊急車輛速度 m/s
        
        etas = {}
        for i, pos in enumerate(intersection_positions):
            if pos > current_position:
                eta = (pos - current_position) / speed
                etas[f'J{i+1}'] = eta
        
        return etas

class WeatherServiceAPI:
    """模擬氣象服務API"""
    
    def get_weather_data(self):
        """模擬氣象資料"""
        weather_conditions = ['sunny', 'cloudy', 'rainy', 'foggy']
        current_condition = np.random.choice(weather_conditions, p=[0.6, 0.25, 0.1, 0.05])
        
        visibility = {
            'sunny': np.random.uniform(8, 10),
            'cloudy': np.random.uniform(6, 9),
            'rainy': np.random.uniform(2, 6),
            'foggy': np.random.uniform(0.5, 3)
        }[current_condition]
        
        return {
            'condition': current_condition,
            'temperature': np.random.uniform(15, 35),
            'humidity': np.random.uniform(40, 90),
            'visibility': visibility,
            'wind_speed': np.random.uniform(0, 15),
            'impact_factor': 1.0 if current_condition == 'sunny' else 
                           0.9 if current_condition == 'cloudy' else
                           0.7 if current_condition == 'rainy' else 0.5
        }

# 初始化Fake API服務
fake_api_services = FakeAPIServices()

# MQTT 客戶端設置 (用於外部系統通訊)
class MQTTHandler:
    """MQTT訊息處理器"""
    
    def __init__(self):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.connected = False
        
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            logger.info("MQTT連接成功")
            # 訂閱相關主題
            client.subscribe("traffic/intersection/+/update")
            client.subscribe("traffic/emergency/alert")
            client.subscribe("traffic/bus/+/position")
        else:
            logger.error(f"MQTT連接失敗: {rc}")
    
    def on_message(self, client, userdata, msg):
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())
            
            if "intersection" in topic:
                intersection_id = topic.split('/')[-2]
                data_manager.add_data('intersection_update', {intersection_id: payload})
            elif "emergency" in topic:
                data_manager.add_data('emergency_event', payload)
            elif "bus" in topic:
                bus_id = topic.split('/')[-2]
                data_manager.add_data('bus_update', {bus_id: payload})
                
        except Exception as e:
            logger.error(f"MQTT訊息處理錯誤: {e}")
    
    def publish_control_command(self, intersection_id, command):
        """發布控制指令"""
        if self.connected:
            topic = f"traffic/control/{intersection_id}"
            self.client.publish(topic, json.dumps(command))

# 初始化MQTT處理器
mqtt_handler = MQTTHandler()

# RESTful API 端點
@app.route('/api/system/status', methods=['GET'])
def get_system_status():
    """獲取系統狀態"""
    return jsonify(data_manager.current_data)

@app.route('/api/intersections', methods=['GET'])
def get_intersections():
    """獲取所有路口資訊"""
    return jsonify(data_manager.current_data['traffic_lights'])

@app.route('/api/intersections/<intersection_id>', methods=['GET'])
def get_intersection(intersection_id):
    """獲取特定路口資訊"""
    intersection_data = data_manager.current_data['traffic_lights'].get(intersection_id)
    if intersection_data:
        return jsonify(intersection_data)
    else:
        return jsonify({'error': 'Intersection not found'}), 404

@app.route('/api/intersections/<intersection_id>/control', methods=['POST'])
def control_intersection(intersection_id):
    """控制路口紅綠燈"""
    try:
        command = request.get_json()
        required_fields = ['phase', 'duration']
        
        if not all(field in command for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # 發送控制指令
        mqtt_handler.publish_control_command(intersection_id, command)
        
        return jsonify({
            'success': True,
            'intersection_id': intersection_id,
            'command': command,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/kpi', methods=['GET'])
def get_kpi():
    """獲取KPI指標"""
    hours = request.args.get('hours', 1, type=int)
    historical_data = data_manager.get_historical_data(hours)
    
    return jsonify({
        'current_kpi': data_manager.current_data['kpi_metrics'],
        'historical_data': historical_data
    })

@app.route('/api/pedestrian', methods=['GET'])
def get_pedestrian_data():
    """獲取行人資料"""
    # 獲取即時模擬資料
    fake_data = fake_api_services.pedestrian_detector.get_detection_data()
    
    # 更新到系統中
    data_manager.add_data('pedestrian_update', fake_data)
    
    return jsonify(fake_data)

@app.route('/api/emergency', methods=['GET'])
def get_emergency_status():
    """獲取緊急車輛狀態"""
    fake_data = fake_api_services.emergency_detector.get_detection_data()
    
    # 如果有緊急事件，更新到系統中
    if fake_data.get('active'):
        data_manager.add_data('emergency_event', fake_data)
    
    return jsonify(fake_data)

@app.route('/api/weather', methods=['GET'])
def get_weather():
    """獲取天氣資訊"""
    return jsonify(fake_api_services.weather_service.get_weather_data())

@app.route('/api/simulation/start', methods=['POST'])
def start_simulation():
    """啟動模擬"""
    # 這裡會整合SUMO模擬
    return jsonify({
        'success': True,
        'message': 'Simulation started',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/simulation/stop', methods=['POST'])
def stop_simulation():
    """停止模擬"""
    return jsonify({
        'success': True,
        'message': 'Simulation stopped',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/ai/decision', methods=['POST'])
def get_ai_decision():
    """獲取AI決策建議"""
    try:
        # 這裡整合AI引擎
        system_state = data_manager.current_data['system_state']
        pedestrian_data = data_manager.current_data['pedestrian_data']
        bus_data = data_manager.current_data['bus_data']
        emergency_data = data_manager.current_data['emergency_data']
        
        # 模擬AI決策回應
        decision = {
            'actions': [0, 1, 0, 2, 0],  # 5個路口的相位建議
            'confidence': 0.87,
            'reasoning': {
                'J1': 'Normal flow, maintain current phase',
                'J2': 'High pedestrian demand, extend green time',
                'J3': 'Coordinate with J2 for green wave',
                'J4': 'Bus priority required',
                'J5': 'Light traffic, standard timing'
            },
            'green_wave_quality': 0.78,
            'estimated_improvement': {
                'travel_time': -15.2,  # % change
                'waiting_time': -22.1,
                'throughput': 8.7
            }
        }
        
        return jsonify(decision)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# WebSocket 事件處理
@socketio.on('connect')
def handle_connect():
    """客戶端連接"""
    logger.info(f"客戶端連接: {request.sid}")
    emit('connected', {'status': 'Connected to traffic system'})
    
    # 發送當前狀態
    emit('data_update', data_manager.current_data)

@socketio.on('disconnect')
def handle_disconnect():
    """客戶端斷線"""
    logger.info(f"客戶端斷線: {request.sid}")

@socketio.on('join_room')
def handle_join_room(data):
    """加入特定房間 (用於分類通知)"""
    room = data.get('room', 'general')
    join_room(room)
    emit('joined_room', {'room': room})

@socketio.on('leave_room')
def handle_leave_room(data):
    """離開房間"""
    room = data.get('room', 'general')
    leave_room(room)
    emit('left_room', {'room': room})

@socketio.on('request_real_time_data')
def handle_real_time_request():
    """處理即時資料請求"""
    emit('data_update', data_manager.current_data)

@socketio.on('control_intersection')
def handle_intersection_control(data):
    """處理路口控制請求"""
    try:
        intersection_id = data.get('intersection_id')
        command = data.get('command')
        
        # 發送控制指令
        mqtt_handler.publish_control_command(intersection_id, command)
        
        emit('control_response', {
            'success': True,
            'intersection_id': intersection_id,
            'command': command
        })
        
    except Exception as e:
        emit('control_response', {'success': False, 'error': str(e)})

# 背景任務：定期更新模擬資料
def background_simulation_updater():
    """背景模擬資料更新器"""
    step = 0
    
    while True:
        try:
            # 模擬系統狀態更新
            system_state = {
                'step': step,
                'timestamp': time.time(),
                'vehicle_count': max(0, int(np.random.poisson(50))),
                'average_speed': max(0, np.random.normal(12, 3)),
                'system_efficiency': max(0, min(1, np.random.beta(4, 2))),
                'total_waiting_vehicles': max(0, int(np.random.poisson(15))),
                'active_intersections': 5
            }
            
            # 模擬路口狀態
            intersection_updates = {}
            for i in range(1, 6):
                intersection_id = f'J{i}'
                intersection_updates[intersection_id] = {
                    'current_phase': np.random.randint(0, 4),
                    'phase_duration': max(15, np.random.normal(35, 10)),
                    'queue_length': max(0, int(np.random.poisson(8))),
                    'waiting_time': max(0, np.random.exponential(20)),
                    'throughput': max(0, int(np.random.poisson(30)))
                }
            
            # 更新資料
            data_manager.add_data('system_state', system_state)
            data_manager.add_data('intersection_update', intersection_updates)
            
            # 獲取fake API資料
            fake_data = fake_api_services.get_all_fake_data()
            if fake_data['pedestrian']:
                data_manager.add_data('pedestrian_update', fake_data['pedestrian'])
            if fake_data['emergency']['active']:
                data_manager.add_data('emergency_event', fake_data['emergency'])
            
            step += 1
            time.sleep(2)  # 每2秒更新一次
            
        except Exception as e:
            logger.error(f"背景更新錯誤: {e}")
            time.sleep(5)

# 啟動背景任務
simulation_thread = threading.Thread(target=background_simulation_updater, daemon=True)
simulation_thread.start()

if __name__ == '__main__':
    logger.info("啟動交通走廊智慧協調系統後端服務")
    logger.info("API 端點:")
    logger.info("  GET  /api/system/status - 系統狀態")
    logger.info("  GET  /api/intersections - 所有路口")
    logger.info("  GET  /api/kpi - KPI指標")
    logger.info("  GET  /api/pedestrian - 行人資料")
    logger.info("  GET  /api/emergency - 緊急車輛")
    logger.info("  POST /api/ai/decision - AI決策")
    
    # 嘗試連接MQTT (可選)
    try:
        mqtt_handler.client.connect("localhost", 1883, 60)
        mqtt_handler.client.loop_start()
    except:
        logger.warning("MQTT服務不可用，繼續使用模擬資料")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)