const express = require("express");
const cors = require("cors");
const bodyParser = require("body-parser");
const fs = require("fs");
const path = require("path");
const pdfParse = require("pdf-parse");
const {
  BedrockRuntimeClient,
  InvokeModelCommand,
} = require("@aws-sdk/client-bedrock-runtime");

const app = express();
app.use(cors());
app.use(bodyParser.json({ limit: "2mb" }));
app.use(express.static(__dirname));

const PORT = process.env.PORT || 3001;
const REGION = process.env.AWS_REGION || "us-east-1";
const MODEL_ID =
  process.env.BEDROCK_MODEL_ID || "anthropic.claude-3-haiku-20240307-v1:0";
const EMBED_MODEL_ID =
  process.env.BEDROCK_EMBED_MODEL_ID || "amazon.titan-embed-text-v1";
const MAX_TOKENS = Number(process.env.MAX_TOKENS || 500);
const TOP_K = Number(process.env.RETRIEVAL_TOP_K || 5);
const DATA_DIR = __dirname;
const CSV_PATH = path.join(DATA_DIR, "long-term-sales-data.csv");
const PDF_PATH = path.join(
  DATA_DIR,
  "Online Garment Tracking and Management System.pdf"
);

const client = new BedrockRuntimeClient({ region: REGION });
let embeddingIndex = [];
let embeddingIndexReady = false;
let embeddingIndexError = null;

async function callBedrock(prompt, context, options = {}) {
  const { maxTokens = MAX_TOKENS, temperature = 0.2 } = options;
  const userText = context ? `${context}\n\nQuestion: ${prompt}` : prompt;

  const payload = {
    anthropic_version: "bedrock-2023-05-31",
    messages: [
      {
        role: "user",
        content: [{ type: "text", text: userText }],
      },
    ],
    max_tokens: maxTokens,
    temperature,
  };

  const command = new InvokeModelCommand({
    modelId: MODEL_ID,
    contentType: "application/json",
    accept: "application/json",
    body: JSON.stringify(payload),
  });

  const response = await client.send(command);
  const decoded = Buffer.from(response.body).toString("utf-8");
  const parsed = JSON.parse(decoded);
  const text = parsed?.content?.[0]?.text?.trim();
  return text || "(no text returned)";
}

async function embedText(text) {
  const payload = { inputText: text };
  const command = new InvokeModelCommand({
    modelId: EMBED_MODEL_ID,
    contentType: "application/json",
    accept: "application/json",
    body: JSON.stringify(payload),
  });
  const response = await client.send(command);
  const decoded = Buffer.from(response.body).toString("utf-8");
  const parsed = JSON.parse(decoded);
  const vector = parsed?.embedding;
  if (!Array.isArray(vector)) {
    throw new Error("No embedding returned");
  }
  return vector;
}

function cosineSimilarity(a, b) {
  const len = Math.min(a.length, b.length);
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function chunkCsv(text, chunkSize = 140) {
  const lines = text.split(/\r?\n/);
  const header = lines[0] || "";
  const dataLines = lines.slice(1);
  const chunks = [];
  for (let i = 0; i < dataLines.length; i += chunkSize) {
    const slice = dataLines.slice(i, i + chunkSize).join("\n");
    chunks.push(`CSV chunk (rows ${i + 1}-${Math.min(i + chunkSize, dataLines.length)}):\n${header}\n${slice}`);
  }
  return chunks;
}

function chunkText(text, maxLen = 1200) {
  const parts = text.split(/\n\s*\n+/).map((p) => p.trim()).filter(Boolean);
  const chunks = [];
  for (const p of parts) {
    if (p.length <= maxLen) {
      chunks.push(p);
    } else {
      for (let i = 0; i < p.length; i += maxLen) {
        chunks.push(p.slice(i, i + maxLen));
      }
    }
  }
  return chunks;
}

async function buildEmbeddingIndex() {
  try {
    const [csv, pdfRaw] = await Promise.all([loadCsvSnippet(), loadPdfText()]);
    const csvChunks = chunkCsv(csv, 120);
    const pdfChunks = chunkText(pdfRaw, 900);
    const corpus = [
      ...csvChunks.map((text, idx) => ({ source: "csv", id: `csv-${idx}`, text })),
      ...pdfChunks.map((text, idx) => ({ source: "pdf", id: `pdf-${idx}`, text })),
    ];
    const vectors = [];
    for (const item of corpus) {
      const vector = await embedText(item.text.slice(0, 2000));
      vectors.push({ ...item, vector });
    }
    embeddingIndex = vectors;
    embeddingIndexReady = true;
    embeddingIndexError = null;
    console.log(`Embedded ${embeddingIndex.length} chunks for retrieval`);
  } catch (err) {
    embeddingIndexError = err;
    embeddingIndexReady = false;
    console.error("Embedding index build failed", err);
  }
}

async function retrieveContext(prompt, k = TOP_K) {
  if (!embeddingIndexReady || embeddingIndex.length === 0) {
    throw embeddingIndexError || new Error("Embedding index not ready");
  }
  const queryVec = await embedText(prompt.slice(0, 4000));
  const scored = embeddingIndex
    .map((item) => ({
      ...item,
      score: cosineSimilarity(queryVec, item.vector),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
  const contextText = scored
    .map((s, i) => `Chunk ${i + 1} [${s.source}]:\n${s.text}`)
    .join("\n\n");
  return contextText;
}

async function loadCsvSnippet(limit = 12000) {
  const text = await fs.promises.readFile(CSV_PATH, "utf8");
  return text.slice(0, limit);
}

async function loadPdfText(limit = 12000) {
  const buffer = await fs.promises.readFile(PDF_PATH);
  const parsed = await pdfParse(buffer);
  const normalized = (parsed.text || "").replace(/\s+/g, " ").trim();
  return normalized.slice(0, limit);
}

async function getFileContext() {
  const [csv, pdf] = await Promise.all([loadCsvSnippet(), loadPdfText()]);
  return { csv, pdf };
}

app.get("/health", (_req, res) => {
  res.json({
    ok: true,
    model: MODEL_ID,
    embedModel: EMBED_MODEL_ID,
    region: REGION,
    indexReady: embeddingIndexReady,
    indexSize: embeddingIndex.length,
    indexError: embeddingIndexError ? String(embeddingIndexError) : null,
  });
});

app.get("/summaries", async (_req, res) => {
  try {
    const { csv, pdf } = await getFileContext();
    const prompt = `
You are an analyst. Given a CSV sample (sales data) and a PDF excerpt (garment tracking system), summarize in JSON with two keys: "purpose" (high-level purpose of these files together) and "status" (current project progress implied by the files). Keep each <= 80 words. Avoid markdown.

CSV sample:
${csv}

PDF excerpt:
${pdf}
`.trim();

    const raw = await callBedrock(prompt, undefined, {
      maxTokens: 400,
      temperature: 0,
    });

    let purpose = "";
    let status = "";
    try {
      const parsed = JSON.parse(raw);
      purpose = parsed?.purpose || "";
      status = parsed?.status || "";
    } catch (_err) {
      const purposeMatch = raw.match(/purpose["']?\s*[:\-]\s*([^\\n]+)/i);
      const statusMatch = raw.match(/status["']?\s*[:\-]\s*([^\\n]+)/i);
      purpose = purposeMatch?.[1]?.trim() || raw;
      status = statusMatch?.[1]?.trim() || raw;
    }

    res.json({
      purpose: purpose || "(no purpose returned)",
      status: status || "(no status returned)",
    });
  } catch (err) {
    console.error("Summary generation failed", err);
    res
      .status(500)
      .json({ error: "Summary generation failed", detail: err.message || String(err) });
  }
});

app.post("/chat", async (req, res) => {
  const { prompt, context } = req.body || {};
  if (!prompt || !prompt.trim()) {
    return res.status(400).json({ error: "Prompt is required" });
  }

  try {
    let retrieval = "";
    try {
      retrieval = await retrieveContext(prompt);
    } catch (retrievalErr) {
      console.warn("Retrieval skipped", retrievalErr);
    }

    const combinedContext = [retrieval, context].filter(Boolean).join("\n\n");
    const concisePrompt = `
Answer concisely (target <= 80 words). If the files do not contain the answer, say that. Cite specific values or sections only as needed.

User question:
${prompt}
`.trim();

    const answer = await callBedrock(concisePrompt, combinedContext);
    res.json({ answer });
  } catch (err) {
    console.error("Bedrock call failed", err);
    res
      .status(500)
      .json({ error: "Bedrock call failed", detail: err.message || String(err) });
  }
});

app.listen(PORT, () => {
  console.log(`Bedrock chat server listening on http://localhost:${PORT}`);
});

buildEmbeddingIndex();
