"""评估指标计算脚本"""
import json
from datetime import datetime

state_path = './finetuned/LoRA_CRM/checkpoint-200/trainer_state.json'
with open(state_path, 'r') as f:
    state = json.load(f)

log_history = state.get('log_history', [])
train_logs = [l for l in log_history if 'loss' in l]
eval_logs = [l for l in log_history if 'eval_loss' in l]

metrics = {}
if train_logs:
    last_train = train_logs[-1]
    metrics['final_loss'] = last_train.get('loss')
    metrics['final_accuracy'] = last_train.get('mean_token_accuracy')
    metrics['total_steps'] = last_train.get('step')

if eval_logs:
    metrics['eval_loss'] = eval_logs[-1]['eval_loss']

print('=' * 50)
print('📊 CRM 微调训练评估报告')
print('=' * 50)
print(f'模型：Qwen3-4B + LoRA')
print(f'训练步数：{metrics["total_steps"]}/500')
print(f'最终 Loss：{metrics["final_loss"]:.4f}')
print(f'最终 Token 准确率：{metrics["final_accuracy"]:.2%}')
print(f'验证 Loss：{metrics["eval_loss"]:.4f}')
print()

print('📈 Loss 变化趋势：')
for l in train_logs:
    print(f'  Step {l["step"]:3d}: loss={l["loss"]:.4f}, acc={l["mean_token_accuracy"]:.2%}')

print()
print('📈 Eval Loss 变化趋势：')
for l in eval_logs:
    print(f'  Step {l["step"]:3d}: eval_loss={l["eval_loss"]:.4f}')

report = {
    'timestamp': datetime.now().isoformat(),
    'model': 'Qwen3-4B',
    'method': 'LoRA',
    'training_metrics': metrics,
    'loss_progression': [{'step': l['step'], 'loss': l['loss'], 'accuracy': l['mean_token_accuracy']} for l in train_logs],
    'eval_progression': [{'step': l['step'], 'eval_loss': l['eval_loss']} for l in eval_logs],
}

with open('./eval_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f'\n评估报告已保存：./eval_report.json')
