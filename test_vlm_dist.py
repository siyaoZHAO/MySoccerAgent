import sys
sys.path.append('/home/zhaosiyao/SoccerAgent/toolbox')
from utils.vlm_distribution import vlm_model, vlm_processor

print("Model successfully loaded!")
print("Model devices:", set(param.device for param in vlm_model.parameters()))
