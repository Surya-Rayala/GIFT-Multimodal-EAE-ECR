from src.processing_engine import ProcessingEngine

expert_folder = ''
session_folder = ''
metric_name = "ENTRANCE_VECTORS"
vmeta_path = ''
engine = ProcessingEngine()
print(engine.compare_expert(metric_name, session_folder, expert_folder, vmeta_path))