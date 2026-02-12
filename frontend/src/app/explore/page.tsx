"use client";

import { useState, useRef, useEffect } from "react";
import {
  Send,
  Loader2,
  Bot,
  User,
  Lightbulb,
  ChevronRight,
  AlertCircle,
  Clock,
  Code,
  Columns3,
  ShieldCheck,
  ShieldAlert,
  ShieldQuestion,
} from "lucide-react";
import { askQuestion, type AskResponse, type ConversationEntry } from "@/lib/api";
import { useSession, type ChatMessage } from "@/lib/store";
import { ResultCard } from "@/components/result-card";

const SUGGESTED_QUESTIONS = [
  "Give me an overview of the data",
  "What are the key correlations?",
  "Are there any outliers?",
  "Describe the numeric columns",
  "What patterns do you see?",
];

function hasResultData(data?: AskResponse): data is AskResponse & { result: Record<string, unknown> } {
  return !!data?.result && Object.keys(data.result).length > 1;
}

function TrustHeader({ data }: { data: AskResponse }) {
  const [showCode, setShowCode] = useState(false);
  const [showColumns, setShowColumns] = useState(false);
  const pct = Math.round(data.confidence * 100);
  const isHigh = pct >= 85;
  const isMod = pct >= 50 && pct < 85;

  return (
    <div className="space-y-2">
      {/* Row 1: skill + trust badge + timing */}
      <div className="flex items-center flex-wrap gap-2 text-xs">
        <span className="px-1.5 py-0.5 rounded bg-slate-100 dark:bg-slate-800 font-mono text-slate-500 dark:text-slate-400">
          {data.skill}
        </span>
        {/* Trust badge */}
        {isHigh ? (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-emerald-50 dark:bg-emerald-950/30 text-emerald-700 dark:text-emerald-400 font-medium">
            <ShieldCheck className="h-3 w-3" />
            {pct}% &middot; High confidence
          </span>
        ) : isMod ? (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-amber-50 dark:bg-amber-950/30 text-amber-700 dark:text-amber-400 font-medium">
            <ShieldQuestion className="h-3 w-3" />
            {pct}% &middot; Moderate confidence
          </span>
        ) : (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-red-50 dark:bg-red-950/30 text-red-700 dark:text-red-400 font-medium">
            <ShieldAlert className="h-3 w-3" />
            {pct}% &middot; Low confidence &mdash; verify results
          </span>
        )}
        <span className="flex items-center gap-0.5 text-slate-400 dark:text-slate-500">
          <Clock className="h-3 w-3" />
          {data.elapsed_seconds.toFixed(1)}s
        </span>
      </div>

      {/* Row 2: reasoning */}
      <p className="text-xs text-slate-500 dark:text-slate-400 italic">
        {data.reasoning}
      </p>

      {/* Row 3: toggle buttons */}
      <div className="flex items-center gap-2">
        {data.code_snippet && (
          <button
            onClick={() => setShowCode((v) => !v)}
            className={`inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium transition-colors ${
              showCode
                ? "bg-violet-100 dark:bg-violet-950/40 text-violet-700 dark:text-violet-300"
                : "bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300"
            }`}
          >
            <Code className="h-3 w-3" />
            {showCode ? "Hide code" : "Show code"}
          </button>
        )}
        {data.columns_used && data.columns_used.length > 0 && (
          <button
            onClick={() => setShowColumns((v) => !v)}
            className={`inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium transition-colors ${
              showColumns
                ? "bg-blue-100 dark:bg-blue-950/40 text-blue-700 dark:text-blue-300"
                : "bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300"
            }`}
          >
            <Columns3 className="h-3 w-3" />
            {data.columns_used.length} columns used
          </button>
        )}
      </div>

      {/* Code snippet panel */}
      {showCode && data.code_snippet && (
        <pre className="text-[11px] leading-relaxed bg-slate-900 dark:bg-slate-950 text-emerald-400 rounded-lg px-4 py-3 overflow-x-auto font-mono">
          {data.code_snippet}
        </pre>
      )}

      {/* Columns used panel */}
      {showColumns && data.columns_used && data.columns_used.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {data.columns_used.map((col) => (
            <span
              key={col}
              className="px-2 py-0.5 rounded-md bg-blue-50 dark:bg-blue-950/30 text-blue-700 dark:text-blue-300 text-xs font-mono"
            >
              {col}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export default function ExplorePage() {
  const { sessionId, filename, columns, exploreMessages, addExploreMessage } = useSession();
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [exploreMessages]);

  if (!sessionId) {
    return (
      <div className="flex items-center justify-center h-[calc(100vh-3.5rem)]">
        <div className="text-center">
          <Bot className="h-12 w-12 text-slate-300 dark:text-slate-600 mx-auto mb-3" />
          <p className="text-slate-500 dark:text-slate-400">
            Upload a dataset first to start exploring.
          </p>
        </div>
      </div>
    );
  }

  async function handleSend(question?: string) {
    const q = (question || input).trim();
    if (!q || loading || !sessionId) return;

    setInput("");
    addExploreMessage({ role: "user", content: q, timestamp: new Date().toISOString() });
    setLoading(true);

    // Build conversation context from last 3 Q&A pairs
    const context: ConversationEntry[] = [];
    const msgs = exploreMessages;
    for (let i = msgs.length - 1; i >= 0 && context.length < 3; i--) {
      const m = msgs[i];
      if (m.role === "assistant" && m.data) {
        // Find the preceding user message
        const userMsg = i > 0 && msgs[i - 1].role === "user" ? msgs[i - 1] : null;
        if (userMsg) {
          const summary = (m.data.narrative || m.content).slice(0, 200);
          context.unshift({ question: userMsg.content.slice(0, 200), summary });
        }
      }
    }

    try {
      const res = await askQuestion(sessionId, q, true, context);
      addExploreMessage({
        role: "assistant",
        content: res.narrative || "Analysis complete.",
        data: res,
        timestamp: new Date().toISOString(),
      });
    } catch (err) {
      addExploreMessage({
        role: "error",
        content: err instanceof Error ? err.message : "Request failed",
        timestamp: new Date().toISOString(),
      });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex h-[calc(100vh-3.5rem)]">
      {/* Sidebar */}
      <aside className="hidden lg:block w-64 border-r border-slate-200 dark:border-slate-800 p-4 overflow-y-auto">
        <h3 className="text-xs font-semibold text-slate-400 dark:text-slate-500 uppercase tracking-wider mb-3">
          Dataset
        </h3>
        <p className="text-sm font-medium text-slate-900 dark:text-white mb-4">
          {filename}
        </p>

        <h3 className="text-xs font-semibold text-slate-400 dark:text-slate-500 uppercase tracking-wider mb-2">
          Columns ({columns.length})
        </h3>
        <ul className="space-y-1 mb-6">
          {columns.map((col) => (
            <li
              key={col.name}
              className="text-xs text-slate-600 dark:text-slate-400 flex items-center justify-between"
            >
              <span className="truncate">{col.name}</span>
              <span className="text-slate-400 dark:text-slate-600 shrink-0 ml-1">
                {col.semantic_type.slice(0, 3)}
              </span>
            </li>
          ))}
        </ul>

        <h3 className="text-xs font-semibold text-slate-400 dark:text-slate-500 uppercase tracking-wider mb-2">
          Try asking
        </h3>
        <ul className="space-y-1">
          {SUGGESTED_QUESTIONS.map((q) => (
            <li key={q}>
              <button
                onClick={() => handleSend(q)}
                disabled={loading}
                className="text-xs text-blue-600 dark:text-blue-400 hover:underline text-left w-full"
              >
                {q}
              </button>
            </li>
          ))}
        </ul>
      </aside>

      {/* Chat area */}
      <div className="flex-1 flex flex-col">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
          {exploreMessages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <Bot className="h-10 w-10 text-slate-300 dark:text-slate-600 mb-3" />
              <p className="text-slate-500 dark:text-slate-400 mb-4">
                Ask a question about your data
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                {SUGGESTED_QUESTIONS.slice(0, 3).map((q) => (
                  <button
                    key={q}
                    onClick={() => handleSend(q)}
                    className="flex items-center gap-1 px-3 py-1.5 rounded-full border border-slate-200 dark:border-slate-700 text-sm text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-900 transition-colors"
                  >
                    <ChevronRight className="h-3 w-3" />
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {exploreMessages.map((msg, i) => (
            <div key={i} className="max-w-3xl mx-auto">
              {msg.role === "user" ? (
                <div className="flex gap-3 justify-end">
                  <div className="bg-blue-600 text-white px-4 py-2 rounded-2xl rounded-br-sm max-w-lg">
                    <p className="text-sm">{msg.content}</p>
                  </div>
                  <div className="w-7 h-7 rounded-full bg-blue-100 dark:bg-blue-900 flex items-center justify-center shrink-0">
                    <User className="h-3.5 w-3.5 text-blue-600 dark:text-blue-300" />
                  </div>
                </div>
              ) : msg.role === "error" ? (
                <div className="flex gap-3">
                  <div className="w-7 h-7 rounded-full bg-red-100 dark:bg-red-900 flex items-center justify-center shrink-0">
                    <AlertCircle className="h-3.5 w-3.5 text-red-600 dark:text-red-300" />
                  </div>
                  <div className="bg-red-50 dark:bg-red-950/30 text-red-600 dark:text-red-400 px-4 py-2 rounded-2xl rounded-bl-sm text-sm">
                    {msg.content}
                  </div>
                </div>
              ) : (
                <div className="flex gap-3">
                  <div className="w-7 h-7 rounded-full bg-slate-100 dark:bg-slate-800 flex items-center justify-center shrink-0">
                    <Bot className="h-3.5 w-3.5 text-slate-600 dark:text-slate-300" />
                  </div>
                  <div className="flex-1 space-y-3">
                    {/* Trust header */}
                    {msg.data && (
                      <TrustHeader data={msg.data} />
                    )}

                    {/* Narrative */}
                    <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl rounded-bl-sm px-4 py-3">
                      <p className="text-sm text-slate-700 dark:text-slate-300 whitespace-pre-wrap">
                        {msg.content}
                      </p>
                    </div>

                    {/* Result data card */}
                    {hasResultData(msg.data) && (
                      <ResultCard result={msg.data.result} skill={msg.data.skill} />
                    )}

                    {/* Key points (clickable — prefills input) */}
                    {msg.data?.key_points && msg.data.key_points.length > 0 && (
                      <div className="bg-blue-50 dark:bg-blue-950/20 border border-blue-100 dark:border-blue-900 rounded-xl px-4 py-3 space-y-1.5">
                        <p className="text-xs font-semibold text-blue-600 dark:text-blue-400 uppercase tracking-wider">
                          Key Points
                        </p>
                        {msg.data.key_points.map((point, j) => (
                          <button
                            key={j}
                            type="button"
                            onClick={() => setInput(`Tell me more about: ${point}`)}
                            className="flex items-start gap-2 text-sm text-slate-700 dark:text-slate-300 w-full text-left cursor-pointer rounded-lg px-2 py-1 -mx-2 hover:bg-blue-100 dark:hover:bg-blue-900/40 transition-colors"
                          >
                            <span className="text-blue-400 dark:text-blue-500 mt-1 shrink-0">&#8226;</span>
                            <span>{point}</span>
                          </button>
                        ))}
                      </div>
                    )}

                    {/* Suggestions as follow-up buttons */}
                    {msg.data?.suggestions &&
                      msg.data.suggestions.length > 0 && (
                        <div className="space-y-1.5">
                          <p className="text-xs font-semibold text-amber-600 dark:text-amber-400 uppercase tracking-wider">
                            Follow up
                          </p>
                          <div className="flex flex-wrap gap-1.5">
                            {msg.data.suggestions.map((s, j) => (
                              <button
                                key={j}
                                onClick={() => handleSend(s)}
                                disabled={loading}
                                className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-medium border border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-950/30 text-amber-700 dark:text-amber-300 hover:bg-amber-100 dark:hover:bg-amber-950/50 transition-colors disabled:opacity-50"
                              >
                                <Lightbulb className="h-3 w-3" />
                                {s}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}


                  </div>
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className="max-w-3xl mx-auto flex gap-3">
              <div className="w-7 h-7 rounded-full bg-slate-100 dark:bg-slate-800 flex items-center justify-center shrink-0">
                <Loader2 className="h-3.5 w-3.5 text-blue-500 animate-spin" />
              </div>
              <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl rounded-bl-sm px-4 py-3">
                <p className="text-sm text-slate-400 dark:text-slate-500">
                  Analyzing...
                </p>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="border-t border-slate-200 dark:border-slate-800 p-4">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSend();
            }}
            className="max-w-3xl mx-auto flex gap-2"
          >
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question about your data..."
              disabled={loading}
              className="flex-1 px-4 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-sm text-slate-900 dark:text-white placeholder:text-slate-400 dark:placeholder:text-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-shadow disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="px-4 py-2.5 rounded-xl bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Send className="h-4 w-4" />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
