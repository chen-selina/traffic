import traci
import os

print(f"當前工作目錄: {os.getcwd()}")
print(f"檔案列表: {os.listdir('.')}")

# 最簡單的SUMO啟動測試
try:
    sumo_cmd = ["sumo", "-c", "configs/corridor.sumocfg", "--begin", "0", "--end", "10"]
    print(f"執行指令: {' '.join(sumo_cmd)}")
    traci.start(sumo_cmd)
    print("SUMO啟動成功!")
    
    for i in range(5):
        traci.simulationStep()
        print(f"Step {i} completed")
    
    traci.close()
    print("測試完成")
    
except Exception as e:
    print(f"錯誤: {e}")