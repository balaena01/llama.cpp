import json, sys, time, urllib.parse, urllib.request

UA = "calib-text-fetch/0.1 (personal research script)"

TITLES = {
    "ja": ["日本", "東京都", "人工知能", "機械学習", "PlayStation 4", "富士山", "江戸時代", "明治維新", "量子力学",
           "光合成", "日本語", "俳句", "将棋", "寿司", "新幹線", "地震", "インターネット", "コンピュータ", "数学",
           "経済学", "源氏物語", "織田信長", "宇宙", "太陽系", "相対性理論", "プログラミング言語", "アニメ", "京都市",
           "第二次世界大戦", "民主主義", "気候変動", "DNA", "ラーメン", "野球", "音楽", "哲学"],
    "zh": ["中国", "北京市", "人工智能", "机器学习", "长城", "唐朝", "汉字", "量子力学", "光合作用", "孔子",
           "互联网", "计算机", "数学", "经济学", "红楼梦", "宇宙", "太阳系", "相对论", "茶", "京剧"],
}

def fetch(lang, title):
    q = urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1, "format": "json",
                                "redirects": 1, "titles": title})
    req = urllib.request.Request(f"https://{lang}.wikipedia.org/w/api.php?{q}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    pages = data.get("query", {}).get("pages", {})
    return "\n".join(p.get("extract", "") for p in pages.values())

out_path = sys.argv[1]
max_chars = {"ja": 6000, "zh": 5000}
with open(out_path, "w", encoding="utf-8") as f:
    for lang, titles in TITLES.items():
        n = 0
        for t in titles:
            try:
                text = fetch(lang, t).strip()
            except Exception as e:
                print("fail", lang, t, e, file=sys.stderr)
                continue
            if not text:
                continue
            text = text[:max_chars[lang]]
            f.write(text + "\n\n")
            n += len(text)
            time.sleep(0.3)
        print(lang, "chars", n)
