import onnx

# 加载模型
model_path = 'capture/local_models/ui_det_v2.onnx'
model = onnx.load(model_path)

# 设置正确的ir_version
model.ir_version = 8  # 根据需要设置正确的ir_version

# 保存修正后的模型
fixed_model_path = 'capture/local_models/ui_det_v2_fixed.onnx'
onnx.save(model, fixed_model_path)

print(f"Fixed model saved to {fixed_model_path}")