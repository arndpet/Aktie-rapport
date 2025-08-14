import os, sys, io, re, orjson, feedparser, pandas as pd
from datetime import datetime, timezone
from dateutil import tz
from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-5")  # funkar även med gpt-4.1
client = OpenAI()

SCHEMA = {
  "type": "object",
  "properties": {
    "extractions": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "company": {"type": "string"},
          "ticker": {"type": "string"},
          "exchange_hint": {"type": "string"},
          "stance": {"type": "string", "enum": ["positive", "negative", "neutral"]},
          "advice_type": {"type": "string", "enum": ["buy", "sell", "hold", "none"]},
          "confidence": {"type": "number", "minimum": 0, "maximum": 1},
          "evidence": {"type": "string"}
        },
        "required": ["company", "stance", "evidence"],
        "additionalProperties": False
      }
    }
  },
  "required": ["extractions"],
  "additionalProperties": False
}

SYSTEM_PROMPT = (
  "Du är en finansanalytiker som extraherar bolag och sentiment ur nyhetstexter. "
  "Returnera ENDAST JSON som följer given JSON Schema. Svenska i friformstext."
)

REPORT_SYSTEM = (
  "Du skriver en saklig svensk marknadsrapport. Struktur: 1) Kort översikt över dagens stämning. "
  "2) Fördjupningar om tongivande bolag med källhänvisningar (t.ex. Källa: Reuters, 14 aug). "
  "3) Avsluta med exakt två topp-10-listor: ‘Topp 10 positivt omnämnda’ och ‘Topp 10 negativt omnämnda’ "
  "(format: TICKER – Bolagsnamn (pos/neg)). Inga andra punktlistor."
)

# --- Läsa RSS ---
feeds = []
with open("feeds.txt", "r", encoding="utf-8") as f:
  feeds = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

articles = []
for url in feeds:
  try:
    feed = feedparser.parse(url)
    for e in feed.entries[:50]:
      title = e.get('title', '')
      summary = re.sub('<[^<]+?>', '', e.get('summary', '') or '')
      link = e.get('link', '')
      published = e.get('published', '') or e.get('updated', '')
      articles.append({
        "source": feed.feed.get('title', 'RSS'),
        "url": link,
        "published_at": published,
        "title": title,
        "snippet": summary,
        "full_text": None
      })
  except Exception as ex:
    print("RSS error:", url, ex, file=sys.stderr)

# Deduplicera: på titel + url
seen = set()
uniq = []
for a in articles:
  key = (a['title'].strip().lower(), a['url'])
  if key in seen:
    continue
  seen.add(key)
  uniq.append(a)
articles = uniq[:200]

# --- Klassificera per artikel ---
records = []
for art in articles:
  content = (f"Källa: {art['source']}\nURL: {art['url']}\nPublicerad: {art['published_at']}\n"
             f"Titel: {art['title']}\nIngress: {art['snippet']}\nText: {art['full_text']}")
  try:
    resp = client.responses.parse(
      model=MODEL,
      input=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content}
      ],
      schema=SCHEMA,
      max_output_tokens=1500
    )
    data = resp.output_parsed or {"extractions": []}
    for ex in data.get("extractions", []):
      ex.update({
        "source": art["source"],
        "url": art["url"],
        "published_at": art["published_at"],
        "title": art["title"]
      })
      records.append(ex)
  except Exception as ex:
    print("LLM error on:", art.get('url'), ex, file=sys.stderr)

if not records:
  print("Inga extraktioner idag. Avslutar.")
  sys.exit(0)

# --- Aggregera ---
df = pd.DataFrame(records)
df["key"] = df["ticker"].fillna("").where(df["ticker"].fillna("") != "", df["company"].str.lower())
agg = df.pivot_table(index=["key"], columns="stance", values="company", aggfunc="count", fill_value=0).reset_index()
for col in ["positive", "negative", "neutral"]:
  if col not in agg:
    agg[col] = 0
agg["net"] = agg["positive"] - agg["negative"]

pos_top = agg.sort_values(["positive", "net"], ascending=[False, False]).head(10)
neg_top = agg.sort_values(["negative", "net"], ascending=[False, True]).head(10)

summary = {
  "generated_at": datetime.now(timezone.utc).isoformat(),
  "positives": pos_top.to_dict(orient="records"),
  "negatives": neg_top.to_dict(orient="records"),
}
with open("top_lists.json", "wb") as f:
  f.write(orjson.dumps(summary))

# --- Skapa slutrapport ---
daily_json = orjson.dumps({
  "by_security": agg.to_dict(orient="records"),
  "examples": df.head(60).to_dict(orient="records"),
}).decode()

report_prompt = [
  {"role": "system", "content": REPORT_SYSTEM},
  {"role": "user", "content": f"Här är dagens sammanställning (JSON): {daily_json}"}
]
rep = client.responses.create(model=MODEL, input=report_prompt, max_output_tokens=2500)
report_text = rep.output_text

# --- Spara med dagens datum i Stockholm ---
stockholm = tz.gettz('Europe/Stockholm')
today = datetime.now(stockholm).strftime('%Y-%m-%d')
os.makedirs('daily_reports', exist_ok=True)
path = os.path.join('daily_reports', f'{today}.md')
with open(path, 'w', encoding='utf-8') as f:
  f.write(report_text)

print("Klart:", path, "och top_lists.json skapade.")
