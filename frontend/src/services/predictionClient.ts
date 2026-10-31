/**
 * The only way the game talks to the CNN.
 *
 * - Every request carries an id; only the newest request may resolve. Older
 *   ones are aborted, and if one still answers, its result is dropped, so a
 *   stale prediction can never advance the game.
 * - Responses are validated field by field. Anything malformed is an error:
 *   the game never fills a gap with a made-up prediction.
 */

export interface ClassScore { className: string; score: number }

export interface PredictionResult {
  requestId: string;
  predictions: ClassScore[];
  uncertain: boolean;
  uncertainBelow: number;
  entropy: number;
  margin: number;
  target: string | null;
  targetScore: number | null;
  targetRank: number | null;
  attention: number[][] | null;
  attentionFocus: string | null;
  modelInput: string;
  modelVersion: string;
  inferenceMs: number;
  roundTripMs: number;
}

export interface ModelInfo {
  modelReady: boolean;
  modelVersion: string | null;
  architecture: string | null;
  parameters: number | null;
  classes: string[];
  difficulty: Record<string, string>;
  temperature: number;
  uncertainBelow: number;
  testMetrics: Record<string, number | null>;
  meanInferenceMs: number | null;
  note: string;
}

export type PredictionErrorKind = 'network' | 'timeout' | 'server' | 'invalid-input' | 'unavailable' | 'malformed' | 'aborted';

export class PredictionError extends Error {
  constructor(public kind: PredictionErrorKind, message: string, public status?: number) { super(message); }
}

type FetchLike = (input: string, init?: RequestInit) => Promise<Response>;

const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

export function parsePrediction(body: any, requestId: string, roundTripMs: number): PredictionResult {
  if (!body || !Array.isArray(body.predictions) || body.predictions.length === 0) {
    throw new PredictionError('malformed', 'The model returned no predictions.');
  }
  const predictions = body.predictions.map((row: any) => {
    if (!row || typeof row.class_name !== 'string' || !isNum(row.score) || row.score < 0 || row.score > 1) {
      throw new PredictionError('malformed', 'A prediction row was malformed.');
    }
    return { className: row.class_name, score: row.score };
  });
  for (let i = 1; i < predictions.length; i++) {
    if (predictions[i].score > predictions[i - 1].score + 1e-9) throw new PredictionError('malformed', 'Predictions were not ranked.');
  }
  if (typeof body.uncertain !== 'boolean' || typeof body.model_version !== 'string' || typeof body.model_input !== 'string') {
    throw new PredictionError('malformed', 'The prediction response was incomplete.');
  }
  if (body.request_id != null && body.request_id !== requestId) {
    throw new PredictionError('malformed', 'The response belongs to a different request.');
  }
  const attention = Array.isArray(body.attention) && body.attention.every((r: unknown) => Array.isArray(r)) ? body.attention as number[][] : null;
  return {
    requestId,
    predictions,
    uncertain: body.uncertain,
    uncertainBelow: isNum(body.uncertain_below) ? body.uncertain_below : 0.5,
    entropy: isNum(body.entropy) ? body.entropy : 0,
    margin: isNum(body.margin) ? body.margin : 0,
    target: typeof body.target === 'string' ? body.target : null,
    targetScore: isNum(body.target_score) ? body.target_score : null,
    targetRank: isNum(body.target_rank) ? body.target_rank : null,
    attention,
    attentionFocus: typeof body.attention_focus === 'string' ? body.attention_focus : null,
    modelInput: body.model_input,
    modelVersion: body.model_version,
    inferenceMs: isNum(body.inference_ms) ? body.inference_ms : 0,
    roundTripMs,
  };
}

export function parseModelInfo(body: any): ModelInfo {
  if (!body || typeof body.model_ready !== 'boolean' || !Array.isArray(body.classes)) {
    throw new PredictionError('malformed', 'Model information was malformed.');
  }
  return {
    modelReady: body.model_ready,
    modelVersion: body.model_version ?? null,
    architecture: body.architecture ?? null,
    parameters: isNum(body.parameters) ? body.parameters : null,
    classes: body.classes.filter((c: unknown) => typeof c === 'string'),
    difficulty: body.difficulty && typeof body.difficulty === 'object' ? body.difficulty : {},
    temperature: isNum(body.temperature) ? body.temperature : 1,
    uncertainBelow: isNum(body.uncertain_below) ? body.uncertain_below : 0.5,
    testMetrics: body.test_metrics && typeof body.test_metrics === 'object' ? body.test_metrics : {},
    meanInferenceMs: isNum(body.mean_inference_ms) ? body.mean_inference_ms : null,
    note: typeof body.note === 'string' ? body.note : '',
  };
}

export class PredictionClient {
  private counter = 0;
  private current: { id: string; controller: AbortController } | null = null;

  constructor(private baseUrl = '', private fetcher: FetchLike = (u, i) => fetch(u, i), private timeoutMs = 12000) {}

  /** Id of the request that is allowed to resolve, if any. */
  get pendingId(): string | null { return this.current?.id ?? null; }

  cancel(): void {
    this.current?.controller.abort();
    this.current = null;
  }

  async modelInfo(): Promise<ModelInfo> {
    let response: Response;
    try {
      response = await this.fetcher(`${this.baseUrl}/api/model`, { headers: { Accept: 'application/json' } });
    } catch {
      throw new PredictionError('network', 'GLITCH’s recognition core is not reachable. Is the backend running?');
    }
    if (!response.ok) throw new PredictionError('server', `Model info failed (HTTP ${response.status}).`, response.status);
    return parseModelInfo(await response.json());
  }

  async predict(pngDataUrl: string, target: string | null): Promise<PredictionResult> {
    this.cancel();
    const id = `r${Date.now().toString(36)}-${++this.counter}`;
    const controller = new AbortController();
    this.current = { id, controller };
    const timer = setTimeout(() => controller.abort(new DOMException('timeout', 'TimeoutError')), this.timeoutMs);
    const started = (typeof performance !== 'undefined' ? performance : Date).now();
    try {
      let response: Response;
      try {
        response = await this.fetcher(`${this.baseUrl}/api/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
          body: JSON.stringify({ image_base64: pngDataUrl, target, request_id: id }),
          signal: controller.signal,
        });
      } catch (error) {
        if (this.current?.id !== id) throw new PredictionError('aborted', 'Superseded by a newer drawing.');
        const reason = (controller.signal as any).reason;
        if (controller.signal.aborted && reason && (reason as DOMException).name === 'TimeoutError') {
          throw new PredictionError('timeout', 'The recognition core took too long to answer.');
        }
        if (controller.signal.aborted) throw new PredictionError('aborted', 'Request cancelled.');
        throw new PredictionError('network', 'Lost the signal to GLITCH’s recognition core. Is the backend running?');
      }
      if (this.current?.id !== id) throw new PredictionError('aborted', 'Superseded by a newer drawing.');
      let body: any = null;
      try { body = await response.json(); } catch { body = null; }
      if (this.current?.id !== id) throw new PredictionError('aborted', 'Superseded by a newer drawing.');
      if (!response.ok) {
        const detail = body && typeof body.detail === 'string' ? body.detail : `HTTP ${response.status}`;
        const kind: PredictionErrorKind = response.status === 422 ? 'invalid-input' : response.status === 503 ? 'unavailable' : 'server';
        throw new PredictionError(kind, detail, response.status);
      }
      const elapsed = (typeof performance !== 'undefined' ? performance : Date).now() - started;
      return parsePrediction(body, id, elapsed);
    } finally {
      clearTimeout(timer);
      if (this.current?.id === id) this.current = null;
    }
  }
}
