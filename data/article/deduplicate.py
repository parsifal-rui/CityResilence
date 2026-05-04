import pandas as pd
import re
from datasketch import MinHash, MinHashLSH

def deduplicate_articles(input_csv: str, output_csv: str, threshold: float = 0.85):
    df = pd.read_csv(input_csv)
    
    # 1. 基础去重：去除所有标点和空白符后完全相同的文章
    df['norm_text'] = df['content'].astype(str).apply(lambda x: re.sub(r'[^\w]', '', x))
    df = df.drop_duplicates(subset=['norm_text']).reset_index(drop=True)
    
    # 2. MinHash LSH 模糊去重：处理增删改写的高度相似文章
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    keep_indices = []
    
    for idx, text in enumerate(df['norm_text']):
        if len(text) < 3:
            keep_indices.append(idx)
            continue
            
        m = MinHash(num_perm=128)
        # 使用 3-gram 提取特征
        for i in range(len(text) - 2):
            m.update(text[i:i+3].encode('utf8'))
            
        # 如果 LSH 桶中没有相似度大于 threshold 的记录，则保留
        if not lsh.query(m):
            lsh.insert(str(idx), m)
            keep_indices.append(idx)
            
    df_final = df.iloc[keep_indices].drop(columns=['norm_text'])
    df_final.to_csv(output_csv, index=False)
    
    print(f"原始数量: {len(df)}, 去重后数量: {len(df_final)}")
    return df_final

if __name__ == "__main__":
    deduplicate_articles(
        input_csv="data/article/articles_cleaned.csv", 
        output_csv="data/article/articles_deduped.csv",
        threshold=0.85  # 调整此值控制相似度阈值，越低去重越狠
    )