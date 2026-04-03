from src.processing_engine import ProcessingEngine

expert_folder = '/Users/surya_rayala/Desktop/Projects/Army-Devcom/GIFT/Archive/Temp/GIFT-Multimodal-EAE-ECR copy/expert_compare_test/expert'
session_folder = '/Users/surya_rayala/Desktop/Projects/Army-Devcom/GIFT/Archive/Temp/GIFT-Multimodal-EAE-ECR copy/expert_compare_test/trainee'
metric_name = "ENTRANCE_VECTORS"
vmeta_path = '/Users/surya_rayala/Desktop/Projects/Army-Devcom/GIFT/GIFT-Multimodal-EAE-main/input/test.vmeta.xml'
engine = ProcessingEngine()
print(engine.compare_expert(metric_name, session_folder, expert_folder, vmeta_path))