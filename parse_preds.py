# parse_preds.py
# 用法:
#   python parse_preds.py --input work_dirs/resnext50_mydata/test_preds.pkl \
#                         --out work_dirs/resnext50_mydata/test_preds.csv \
#                         --excel work_dirs/resnext50_mydata/test_preds.xlsx \
#                         --confusion work_dirs/resnext50_mydata/confusion_matrix.csv
#
# 说明:
# - 解析 mmpretrain 的 --out 生成的 pkl（通常是 --out-item pred）
# - 导出每张图片: 文件路径(img_path), 预测top1标签名/索引, 真值标签名/索引, 置信度
# - 若无法拿到 classes 名称, 会回退为数字标签
# - 额外增加一列 gt_folder：从 img_path 的上一级目录名提取，与你数据集的真实文件夹一致
# - 可选导出混淆矩阵 CSV

import argparse
import pickle
import os
import numpy as np
import pandas as pd

def to_int(x):
    # 将各种 tensor/list/ndarray/标量 转成 int
    if x is None:
        return None
    if hasattr(x, 'item'):
        try:
            return int(x.item())
        except Exception:
            pass
    if isinstance(x, (list, tuple, np.ndarray)):
        return int(np.array(x).astype(np.int64).ravel()[0])
    return int(x)

def to_float(x):
    if x is None:
        return None
    if hasattr(x, 'item'):
        try:
            return float(x.item())
        except Exception:
            pass
    if isinstance(x, (list, tuple, np.ndarray)):
        return float(np.array(x, dtype=np.float32).ravel()[0])
    try:
        return float(x)
    except Exception:
        return None

def get_classes_from_anywhere(obj):
    """尽量从预测结果里挖 classes 名称列表。"""
    # 1) 顶层字典里
    for key in ['metainfo', 'meta', 'data_metas', 'classes']:
        if isinstance(obj, dict) and key in obj:
            if key == 'classes' and isinstance(obj[key], (list, tuple)):
                return list(obj[key])
            mi = obj.get(key)
            if isinstance(mi, dict):
                for k2 in ['classes', 'CLASSES', 'labels']:
                    if k2 in mi and isinstance(mi[k2], (list, tuple)):
                        return list(mi[k2])

    # 2) 列表/迭代每个样本里
    if isinstance(obj, (list, tuple)):
        for it in obj:
            if isinstance(it, dict):
                for key in ['metainfo', 'meta', 'data_metas']:
                    if key in it and isinstance(it[key], dict):
                        mi = it[key]
                        for k2 in ['classes', 'CLASSES', 'labels']:
                            if k2 in mi and isinstance(mi[k2], (list, tuple)):
                                return list(mi[k2])
                # 有些结果把 classes 放到 it['inputs']['metainfo']
                if 'inputs' in it and isinstance(it['inputs'], dict):
                    for key in ['metainfo', 'meta']:
                        if key in it['inputs'] and isinstance(it['inputs'][key], dict):
                            mi = it['inputs'][key]
                            for k2 in ['classes', 'CLASSES', 'labels']:
                                if k2 in mi and isinstance(mi[k2], (list, tuple)):
                                    return list(mi[k2])
    return None

def top1_from_scores(scores):
    """给一条样本的 score 向量，返回 (best_idx, best_score)"""
    if scores is None:
        return None, None
    arr = np.array(scores, dtype=np.float32).ravel()
    if arr.size == 0:
        return None, None
    idx = int(arr.argmax())
    return idx, float(arr[idx])

def extract_one(item):
    """
    尽量兼容不同结构，把一条预测解析成:
    {
      'img_path': str or None,
      'pred_idx': int or None,
      'gt_idx': int or None,
      'conf': float or None,
      'scores': list/ndarray or None
    }
    """
    img_path = None
    pred_idx = None
    gt_idx = None
    conf = None
    scores = None

    # 常见字段优先（mmpretrain 通常已展平成 dict）
    if isinstance(item, dict):
        # 文件路径
        for k in ['img_path', 'img_paths', 'filename', 'ori_filename', 'img']:
            if k in item:
                v = item[k]
                img_path = v if isinstance(v, str) else (v[0] if isinstance(v, (list, tuple)) and v else None)
                if img_path:
                    break
        # 嵌套 inputs 里也可能带路径/元信息
        if img_path is None and 'inputs' in item and isinstance(item['inputs'], dict):
            for k in ['img_path', 'filename', 'ori_filename']:
                if k in item['inputs']:
                    img_path = item['inputs'][k]
                    break

        # 预测标签索引
        for k in ['pred_label', 'pred_idx', 'label', 'pred_class']:
            if k in item:
                pred_idx = to_int(item[k])
                break

        # 分数向量
        for k in ['pred_scores', 'scores', 'score']:
            if k in item:
                scores = item[k]
                break

        # 单独的 top1 置信度
        if conf is None:
            for k in ['pred_score', 'confidence', 'prob']:
                if k in item:
                    conf = to_float(item[k])
                    break

        # 真值
        for k in ['gt_label', 'gt_idx', 'target', 'label']:
            if k in item and gt_idx is None:
                gt_idx = to_int(item[k])
                break

    else:
        # DataSample-like 对象，尽量 getattr
        for name in ['img_path', 'filename', 'ori_filename']:
            if hasattr(item, name):
                v = getattr(item, name)
                if isinstance(v, str):
                    img_path = v
                    break

        # pred / scores
        for name in ['pred_label', 'pred_idx', 'label']:
            if hasattr(item, name):
                pred_idx = to_int(getattr(item, name))
                break
        for name in ['pred_scores', 'scores', 'score']:
            if hasattr(item, name):
                scores = getattr(item, name)
                break
        for name in ['pred_score', 'confidence', 'prob']:
            if hasattr(item, name):
                conf = to_float(getattr(item, name))
                break
        for name in ['gt_label', 'gt_idx', 'target', 'label']:
            if hasattr(item, name):
                gt_idx = to_int(getattr(item, name))
                break

    # 如果 pred_idx 没有，但有 scores，则用 scores 求 top1
    if pred_idx is None and scores is not None:
        pred_idx, conf_from_scores = top1_from_scores(scores)
        if conf is None:
            conf = conf_from_scores

    return img_path, pred_idx, gt_idx, conf, scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='test.py --out 生成的 pkl 文件路径')
    ap.add_argument('--out', required=True, help='要保存的 CSV 路径')
    ap.add_argument('--excel', default=None, help='可选：另存为 Excel 路径（.xlsx）')
    ap.add_argument('--confusion', default=None, help='可选：输出混淆矩阵 CSV 路径')
    args = ap.parse_args()

    with open(args.input, 'rb') as f:
        data = pickle.load(f)

    # 兼容: 有时是 {'predictions': [...]} 或 {'results': [...]} 或直接 list
    if isinstance(data, dict):
        for key in ['predictions', 'results', 'preds', 'data', 'items']:
            if key in data and isinstance(data[key], (list, tuple)):
                records = data[key]
                break
        else:
            lists = [(k, v) for k, v in data.items() if isinstance(v, (list, tuple))]
            if lists:
                lists.sort(key=lambda kv: len(kv[1]), reverse=True)
                records = lists[0][1]
            else:
                raise ValueError('未找到预测列表，请检查 pkl 结构')
    else:
        records = data

    classes = get_classes_from_anywhere(data)

    def name_or_idx(idx):
        # 若没有 classes，先用数字字符串
        if idx is None:
            return None
        if classes and 0 <= idx < len(classes):
            return classes[idx]
        return str(idx)

    rows = []
    for it in records:
        img_path, pred_idx, gt_idx, conf, scores = extract_one(it)
        rows.append({
            'img_path': img_path,
            'pred_idx': pred_idx,
            'pred_name': name_or_idx(pred_idx),
            'gt_idx': gt_idx,
            'gt_name': name_or_idx(gt_idx),
            'confidence': conf
        })

    df = pd.DataFrame(rows)

    # 从路径提取简短文件名 & 上一级目录名（gt_folder）
    if 'img_path' in df.columns:
        df['filename'] = df['img_path'].apply(lambda p: os.path.basename(p) if isinstance(p, str) else None)
        df['gt_folder'] = df['img_path'].apply(
            lambda p: os.path.basename(os.path.dirname(p)) if isinstance(p, str) else None
        )
        # 列顺序更友好
        cols = ['filename', 'img_path',
                'gt_folder', 'gt_name', 'gt_idx',
                'pred_name', 'pred_idx', 'confidence']
        df = df[[c for c in cols if c in df.columns]]

    # 导出
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False, encoding='utf-8-sig')
    print(f'[OK] CSV saved to: {args.out}')
    if args.excel:
        df.to_excel(args.excel, index=False)
        print(f'[OK] Excel saved to: {args.excel}')

    # 可选：混淆矩阵
    if args.confusion:
        # 仅在标签都存在时计算
        if df['pred_idx'].notna().all() and df['gt_idx'].notna().all():
            y_true = df['gt_idx'].astype(int).values
            y_pred = df['pred_idx'].astype(int).values
            n = max(int(y_true.max()), int(y_pred.max())) + 1
            cm = np.zeros((n, n), dtype=int)
            for t, p in zip(y_true, y_pred):
                cm[t, p] += 1
            # 构造带类名的 DataFrame
            if classes and len(classes) >= n:
                idx_names = classes[:n]
                col_names = classes[:n]
            else:
                idx_names = [f'gt_{i}' for i in range(n)]
                col_names = [f'pred_{i}' for i in range(n)]
            cm_df = pd.DataFrame(cm, index=idx_names, columns=col_names)
            cm_df.to_csv(args.confusion, encoding='utf-8-sig')
            print(f'[OK] Confusion matrix saved to: {args.confusion}')
            acc = (y_true == y_pred).mean()
            print(f'[INFO] Overall accuracy (by rows): {acc:.4f}')
        else:
            print('[WARN] 有缺失标签，未生成混淆矩阵。')

if __name__ == '__main__':
    main()
