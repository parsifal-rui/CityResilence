import re
import pandas as pd
from playwright.sync_api import sync_playwright
import time

DATE_PATTERN = re.compile(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}')

def extract_urls(excel_path: str):
    df = pd.read_excel(excel_path)
    urls = df['原文/评论链接'].dropna().unique().tolist()
    return [url for url in urls if 'toutiao.com' in str(url)]

def fetch_article(page, url: str):
    try:
        page.goto(url, timeout=10000, wait_until='domcontentloaded')
        page.wait_for_selector('p', timeout=5000)
        time.sleep(0.5)
        
        # 提取标题
        try:
            title = page.locator('h1').first.inner_text()
        except:
            title = ''
        
        # 提取正文
        content = ''
        selectors = [
            '.article-content p',
            '.article-meta p',
            'article p',
        ]
        
        for selector in selectors:
            try:
                paras = page.locator(selector).all()
                texts = [p.inner_text() for p in paras if p.inner_text().strip()]
                content = '\n'.join(texts)
                if content:
                    break
            except:
                continue
        
        publish_time = ''
        for sel in ['span.time', '.article-meta span', 'span']:
            try:
                els = page.locator(sel).all()
                for el in els[:20]:
                    t = el.inner_text().strip()
                    m = DATE_PATTERN.search(t)
                    if m:
                        publish_time = m.group(0)
                        break
                if publish_time:
                    break
            except:
                continue
        
        return {
            'url': url,
            'title': title,
            'content': content,
            'publish_time': publish_time,
            'status': 'success' if content else 'empty'
        }
    except Exception as e:
        return {'url': url, 'title': '', 'content': '', 'publish_time': '', 'status': f'failed: {str(e)}'}

def crawl_all(excel_path: str, output_path: str):
    urls = extract_urls(excel_path)
    print(f"提取到 {len(urls)} 个URL")
    
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        page = context.new_page()
        
        for i, url in enumerate(urls, 1):
            print(f"[{i}/{len(urls)}] {url}")
            result = fetch_article(page, url)
            results.append(result)
            print(f"  -> {result['status']}")
            
            if i % 10 == 0:
                pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8-sig')
        
        browser.close()
    
    df_result = pd.DataFrame(results)
    df_result.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    success = (df_result['status'] == 'success').sum()
    print(f"\n完成！成功: {success}/{len(urls)}")

if __name__ == '__main__':
    crawl_all(
        excel_path='灾害新闻广东_240801_240930.xlsx',
        output_path='articles_playwright.csv'
    )
