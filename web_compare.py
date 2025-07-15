import os
import time
import torch
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify
from dataset.dezh import TranslationDataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 配置
base_dir = "./train_process/distillation-dezh"
student_model_path = Path(base_dir + "/distillation_checkpoints/best_student.pt")
teacher_model_path = Path("./train_process/transformer-dezh/transformer_checkpoints/best.pt")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_dir = "data/de-zh/de-zh.txt"

# 初始化数据集
print("加载数据集...")
dataset = TranslationDataset(data_dir)
max_seq_length = 42

# 预加载模型
print("加载学生模型...")
student_model = torch.load(student_model_path, map_location=device)
student_model.to(device)
student_model.eval()

print("加载教师模型...")
teacher_model = torch.load(teacher_model_path, map_location=device)
teacher_model.to(device)
teacher_model.eval()

# 测试句子
TEST_SENTENCES = [
    "Am Anfang schuf Gott Himmel und Erde.",
    "Und Gott sprach: Es werde Licht! Und es ward Licht.",
    "Du sollst nicht töten.",
    "Denn also hat Gott die Welt geliebt, dass er seinen eingeborenen Sohn gab.",
    "Ich bin der Weg und die Wahrheit und das Leben.",
    "Liebe deinen Nächsten wie dich selbst."
]

# 参考翻译（如有，可补充）
REFERENCE_TRANSLATIONS = [
    "起 初 神 創 造 天 地",
    "神說 、 要 有 光 、 就 有 了 光",
    "不可殺人",
    "因 为 神 爱 世 人 、 将 他 的 独 生 子 赐 给 世 人",
    "我 就是 道路 、 真理 、 生命 ",
    "愛人 如 己",
]

# 翻译函数
def calc_repeat_rate(text):
    tokens = text.split()
    if not tokens: return 0
    repeat = sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i-1])
    return repeat / len(tokens)

def calc_token_diff(t1, t2):
    # 统计两个token序列的不同token数量
    s1 = set(t1.split())
    s2 = set(t2.split())
    return len(s1.symmetric_difference(s2))

def translate(model, src: str):
    src_tokens = [0] + dataset.de_vocab(dataset.de_tokenizer(src)) + [1]
    src_tensor = torch.tensor(src_tokens).unsqueeze(0).to(device)
    tgt_tensor = torch.tensor([[0]]).to(device)
    generated_tokens = []
    repeat_count = 0
    last_token = None
    with torch.no_grad():
        for _ in range(max_seq_length):
            out = model(src_tensor, tgt_tensor)
            predict = model.predictor(out[:, -1])
            next_token = torch.argmax(predict, dim=1)
            token_id = next_token.item()
            # 检查重复
            if token_id == last_token:
                repeat_count += 1
            else:
                repeat_count = 0
            last_token = token_id
            if repeat_count >= 5:  # 连续5次重复就break
                break
            tgt_tensor = torch.cat([tgt_tensor, next_token.unsqueeze(0)], dim=1)
            if token_id == 1:  # <eos>
                break
    tgt_tokens = tgt_tensor.squeeze().tolist()
    translated = ' '.join(dataset.zh_vocab.lookup_tokens(tgt_tokens))
    return translated.replace("<s>", "").replace("</s>", "").strip()

# BLEU分数计算
def calc_bleu(reference, candidate):
    if reference is None or reference.strip() == "":
        return None
    ref_tokens = list(reference.replace(' ', ''))
    cand_tokens = list(candidate.replace(' ', ''))
    smoothie = SmoothingFunction().method4
    return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie)

# 获取模型大小
def get_model_size(path):
    size_bytes = Path(path).stat().st_size
    return size_bytes / (1024 * 1024)

# Flask应用
app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string(TEMPLATE)

@app.route('/compare', methods=['GET'])
def compare():
    results = []
    bleu_teacher_list = []
    bleu_student_list = []
    teacher_time_list = []
    student_time_list = []
    for idx, src in enumerate(TEST_SENTENCES):
        # 教师模型
        t0 = time.time()
        teacher_trans = translate(teacher_model, src)
        t1 = time.time()
        teacher_time = t1 - t0
        # 学生模型
        t0 = time.time()
        student_trans = translate(student_model, src)
        t1 = time.time()
        student_time = t1 - t0
        # BLEU
        ref = REFERENCE_TRANSLATIONS[idx] if REFERENCE_TRANSLATIONS[idx] else teacher_trans
        bleu_teacher = calc_bleu(ref, teacher_trans)
        bleu_student = calc_bleu(ref, student_trans)
        bleu_teacher_list.append(bleu_teacher if bleu_teacher is not None else 0)
        bleu_student_list.append(bleu_student if bleu_student is not None else 0)
        teacher_time_list.append(teacher_time)
        student_time_list.append(student_time)
        # 翻译长度
        teacher_len = len(teacher_trans.split())
        student_len = len(student_trans.split())
        # 重复率
        teacher_repeat = calc_repeat_rate(teacher_trans)
        student_repeat = calc_repeat_rate(student_trans)
        # token差异
        token_diff = calc_token_diff(teacher_trans, student_trans)
        results.append({
            'src': src,
            'teacher_trans': teacher_trans,
            'student_trans': student_trans,
            'teacher_time': teacher_time,
            'student_time': student_time,
            'bleu_teacher': bleu_teacher,
            'bleu_student': bleu_student,
            'teacher_len': teacher_len,
            'student_len': student_len,
            'teacher_repeat': teacher_repeat,
            'student_repeat': student_repeat,
            'token_diff': token_diff
        })
    # 模型大小
    teacher_size = get_model_size(teacher_model_path)
    student_size = get_model_size(student_model_path)
    compression_ratio = teacher_size / student_size if student_size > 0 else 0
    return jsonify({
        'results': results,
        'teacher_size': teacher_size,
        'student_size': student_size,
        'compression_ratio': compression_ratio,
        'bleu_teacher_list': bleu_teacher_list,
        'bleu_student_list': bleu_student_list,
        'teacher_time_list': teacher_time_list,
        'student_time_list': student_time_list
    })

TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <title>模型对比可视化</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://fonts.googleapis.com/css?family=Roboto:400,700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Roboto', '微软雅黑', Arial, sans-serif;
            margin: 0;
            background: #f6f8fa;
        }
        .header {
            background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
            color: #fff;
            padding: 32px 0 20px 40px;
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: 2px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }
        .container {
            max-width: 1200px;
            margin: 30px auto 0 auto;
            background: #fff;
            border-radius: 16px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.08);
            padding: 32px 40px 40px 40px;
        }
        .charts-row {
            display: flex;
            gap: 40px;
            margin-bottom: 40px;
            flex-wrap: wrap;
        }
        .chart-card {
            background: #f9fafb;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            padding: 18px 18px 8px 18px;
            flex: 1 1 350px;
            min-width: 350px;
            margin-bottom: 10px;
        }
        .refresh-btn {
            background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
            color: #fff;
            border: none;
            border-radius: 8px;
            padding: 10px 28px;
            font-size: 1.1rem;
            font-weight: 700;
            cursor: pointer;
            margin-bottom: 30px;
            box-shadow: 0 2px 8px rgba(31,119,180,0.08);
            transition: background 0.3s, box-shadow 0.3s;
        }
        .refresh-btn:hover {
            background: linear-gradient(90deg, #ff7f0e 0%, #1f77b4 100%);
            box-shadow: 0 4px 16px rgba(255,127,14,0.12);
        }
        .big-number {
            font-size: 2.5rem;
            font-weight: 700;
            color: #ff7f0e;
            margin-bottom: 10px;
        }
        .big-label {
            font-size: 1.1rem;
            color: #888;
            margin-bottom: 30px;
        }
        table {
            border-collapse: separate;
            border-spacing: 0;
            width: 100%;
            margin-top: 30px;
            background: #fff;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }
        th, td {
            padding: 10px 8px;
            text-align: center;
        }
        th {
            background: #f2f4f8;
            font-weight: 700;
        }
        tr:nth-child(even) {
            background: #f9fafb;
        }
        tr:hover {
            background: #eaf6ff;
        }
    </style>
</head>
<body>
    <div class="header">教师模型 vs 学生模型 可视化对比</div>
    <div class="container">
        <button class="refresh-btn" onclick="loadData()">刷新对比</button>
        <div class="big-number" id="compress_ratio"></div>
        <div class="big-label">模型压缩率（教师模型/学生模型）</div>
        <div class="charts-row">
            <div class="chart-card"><div id="size_chart" style="width:100%;height:300px;"></div></div>
            <div class="chart-card"><div id="speed_chart" style="width:100%;height:300px;"></div></div>
        </div>
        <div class="charts-row">
            <div class="chart-card"><div id="len_chart" style="width:100%;height:300px;"></div></div>
            <div class="chart-card"><div id="repeat_chart" style="width:100%;height:300px;"></div></div>
        </div>
        <div class="chart-card" style="margin-bottom:30px;"><div id="bleu_chart" style="width:100%;height:320px;"></div></div>
        <div class="chart-card" style="margin-bottom:30px;"><div id="diff_chart" style="width:100%;height:320px;"></div></div>
        <div class="charts-row">
            <div class="chart-card"><div id="speed_box" style="width:100%;height:300px;"></div></div>
            <div class="chart-card"><div id="bleu_box" style="width:100%;height:300px;"></div></div>
        </div>
        <div id="table"></div>
    </div>
<script>
function loadData() {
    fetch('/compare').then(r=>r.json()).then(data=>{
        // 压缩率
        document.getElementById('compress_ratio').innerText = data.compression_ratio.toFixed(2) + 'x';
        // 模型大小
        let sizeData = [{
            x:['教师模型','学生模型'],
            y:[data.teacher_size, data.student_size],
            type:'bar',
            text:[data.teacher_size.toFixed(2)+'MB', data.student_size.toFixed(2)+'MB'],
            textposition:'auto',
            marker:{color:['#1f77b4','#ff7f0e']}
        }];
        Plotly.newPlot('size_chart', sizeData, {title:'模型大小(MB)', paper_bgcolor:'#f9fafb', plot_bgcolor:'#f9fafb'});
        // 推理速度
        let avg_teacher = data.results.reduce((a,b)=>a+b.teacher_time,0)/data.results.length;
        let avg_student = data.results.reduce((a,b)=>a+b.student_time,0)/data.results.length;
        let speedData = [{
            x:['教师模型','学生模型'],
            y:[avg_teacher, avg_student],
            type:'bar',
            text:[avg_teacher.toFixed(3)+'s', avg_student.toFixed(3)+'s'],
            textposition:'auto',
            marker:{color:['#1f77b4','#ff7f0e']}
        }];
        Plotly.newPlot('speed_chart', speedData, {title:'平均推理时间(秒)', paper_bgcolor:'#f9fafb', plot_bgcolor:'#f9fafb'});
        // 翻译长度对比
        let lenData = [
            {x:data.results.map((_,i)=>'句子'+(i+1)), y:data.results.map(r=>r.teacher_len), name:'教师模型', type:'bar', marker:{color:'#1f77b4'}},
            {x:data.results.map((_,i)=>'句子'+(i+1)), y:data.results.map(r=>r.student_len), name:'学生模型', type:'bar', marker:{color:'#ff7f0e'}}
        ];
        Plotly.newPlot('len_chart', lenData, {barmode:'group', title:'翻译长度对比', paper_bgcolor:'#f9fafb', plot_bgcolor:'#f9fafb'});
        // 重复率对比
        let repeatData = [
            {x:data.results.map((_,i)=>'句子'+(i+1)), y:data.results.map(r=>r.teacher_repeat), name:'教师模型', type:'bar', marker:{color:'#1f77b4'}},
            {x:data.results.map((_,i)=>'句子'+(i+1)), y:data.results.map(r=>r.student_repeat), name:'学生模型', type:'bar', marker:{color:'#ff7f0e'}}
        ];
        Plotly.newPlot('repeat_chart', repeatData, {barmode:'group', title:'重复率对比', paper_bgcolor:'#f9fafb', plot_bgcolor:'#f9fafb'});
        // BLEU分数
        let bleu_teacher = data.results.map(r=>r.bleu_teacher||0);
        let bleu_student = data.results.map(r=>r.bleu_student||0);
        let bleuData = [
            {x:data.results.map((_,i)=>'句子'+(i+1)), y:bleu_teacher, name:'教师模型', type:'bar', marker:{color:'#1f77b4'}},
            {x:data.results.map((_,i)=>'句子'+(i+1)), y:bleu_student, name:'学生模型', type:'bar', marker:{color:'#ff7f0e'}}
        ];
        Plotly.newPlot('bleu_chart', bleuData, {barmode:'group', title:'BLEU分数对比', paper_bgcolor:'#f9fafb', plot_bgcolor:'#f9fafb'});
        // token差异热力图
        let diffData = [{
            z: [data.results.map(r=>r.token_diff)],
            x: data.results.map((_,i)=>'句子'+(i+1)),
            y: ['token差异数'],
            type: 'heatmap',
            colorscale: 'YlOrRd',
            showscale: true
        }];
        Plotly.newPlot('diff_chart', diffData, {title:'翻译token差异热力图', paper_bgcolor:'#f9fafb', plot_bgcolor:'#f9fafb'});
        // 推理速度分布
        let speedBox = [
            {y:data.teacher_time_list, name:'教师模型', type:'box', marker:{color:'#1f77b4'}},
            {y:data.student_time_list, name:'学生模型', type:'box', marker:{color:'#ff7f0e'}}
        ];
        Plotly.newPlot('speed_box', speedBox, {title:'推理速度分布', paper_bgcolor:'#f9fafb', plot_bgcolor:'#f9fafb'});
        // BLEU分数分布
        let bleuBox = [
            {y:data.bleu_teacher_list, name:'教师模型', type:'box', marker:{color:'#1f77b4'}},
            {y:data.bleu_student_list, name:'学生模型', type:'box', marker:{color:'#ff7f0e'}}
        ];
        Plotly.newPlot('bleu_box', bleuBox, {title:'BLEU分数分布', paper_bgcolor:'#f9fafb', plot_bgcolor:'#f9fafb'});
        // 表格
        let html = '<table><tr><th>原始德语</th><th>教师模型翻译</th><th>学生模型翻译</th><th>教师推理(s)</th><th>学生推理(s)</th><th>教师BLEU</th><th>学生BLEU</th><th>教师长度</th><th>学生长度</th><th>教师重复率</th><th>学生重复率</th><th>token差异</th></tr>';
        data.results.forEach(r=>{
            html += `<tr><td>${r.src}</td><td>${r.teacher_trans}</td><td>${r.student_trans}</td><td>${r.teacher_time.toFixed(3)}</td><td>${r.student_time.toFixed(3)}</td><td>${r.bleu_teacher? r.bleu_teacher.toFixed(3):'-'}</td><td>${r.bleu_student? r.bleu_student.toFixed(3):'-'}</td><td>${r.teacher_len}</td><td>${r.student_len}</td><td>${(r.teacher_repeat*100).toFixed(1)}%</td><td>${(r.student_repeat*100).toFixed(1)}%</td><td>${r.token_diff}</td></tr>`;
        });
        html += '</table>';
        document.getElementById('table').innerHTML = html;
    });
}
loadData();
</script>
</body>
</html>
'''

if __name__ == '__main__':
    import webbrowser
    import threading
    def open_browser():
        webbrowser.open('http://127.0.0.1:5000')
    threading.Timer(1, open_browser).start()
    app.run(debug=False, port=5000) 