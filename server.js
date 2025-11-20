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
const MAX_TOKENS = Number(process.env.MAX_TOKENS || 500);
const DATA_DIR = __dirname;
const CSV_PATH = path.join(DATA_DIR, "long-term-sales-data.csv");
const PDF_PATH = path.join(
  DATA_DIR,
  "Online Garment Tracking and Management System.pdf"
);

const client = new BedrockRuntimeClient({ region: REGION });

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
  res.json({ ok: true, model: MODEL_ID, region: REGION });
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
    const answer = await callBedrock(prompt, context);
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
