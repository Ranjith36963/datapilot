"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { useRouter } from "next/navigation";
import {
  Upload,
  FileSpreadsheet,
  Loader2,
  ArrowRight,
  Columns3,
  Rows3,
  AlertCircle,
} from "lucide-react";
import { uploadFile, type ColumnInfo } from "@/lib/api";
import { useSession } from "@/lib/store";
import { DataTable } from "@/components/data-table";

export default function UploadPage() {
  const router = useRouter();
  const { setSession, sessionId, filename, shape, columns, preview } =
    useSession();

  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (!file) return;

      setUploading(true);
      setError(null);

      try {
        const res = await uploadFile(file);
        setSession({
          sessionId: res.session_id,
          filename: res.filename,
          shape: res.shape,
          columns: res.columns,
          preview: res.preview,
        });
      } catch (err) {
        setError(err instanceof Error ? err.message : "Upload failed");
      } finally {
        setUploading(false);
      }
    },
    [setSession]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "text/csv": [".csv"],
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [
        ".xlsx",
      ],
      "application/vnd.ms-excel": [".xls"],
      "application/json": [".json"],
      "application/x-parquet": [".parquet"],
    },
    maxFiles: 1,
    disabled: uploading,
  });

  const hasData = sessionId && columns.length > 0;

  return (
    <div className="max-w-5xl mx-auto px-4 py-12">
      {/* Hero */}
      <div className="text-center mb-10">
        <h1 className="text-4xl font-bold text-slate-900 dark:text-white mb-3">
          Analyze your data with AI
        </h1>
        <p className="text-lg text-slate-500 dark:text-slate-400 max-w-xl mx-auto">
          Upload a dataset and ask questions in natural language.
          DataPilot routes your questions to 81+ analysis skills.
        </p>
      </div>

      {/* Upload zone */}
      <div
        {...getRootProps()}
        className={`
          relative border-2 border-dashed rounded-xl p-12 text-center
          cursor-pointer transition-all duration-200
          ${
            isDragActive
              ? "border-blue-500 bg-blue-50 dark:bg-blue-950/30"
              : "border-slate-300 dark:border-slate-700 hover:border-blue-400 dark:hover:border-blue-600 hover:bg-slate-50 dark:hover:bg-slate-900/50"
          }
          ${uploading ? "pointer-events-none opacity-60" : ""}
        `}
      >
        <input {...getInputProps()} />
        {uploading ? (
          <div className="flex flex-col items-center gap-3">
            <Loader2 className="h-10 w-10 text-blue-500 animate-spin" />
            <p className="text-slate-600 dark:text-slate-300 font-medium">
              Processing your file...
            </p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3">
            <Upload className="h-10 w-10 text-slate-400 dark:text-slate-500" />
            <p className="text-slate-600 dark:text-slate-300 font-medium">
              {isDragActive
                ? "Drop your file here"
                : "Drag & drop a file, or click to browse"}
            </p>
            <p className="text-sm text-slate-400 dark:text-slate-500">
              CSV, Excel, JSON, or Parquet — up to 100 MB
            </p>
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="mt-4 flex items-center gap-2 p-3 rounded-lg bg-red-50 dark:bg-red-950/30 text-red-600 dark:text-red-400 text-sm">
          <AlertCircle className="h-4 w-4 shrink-0" />
          {error}
        </div>
      )}

      {/* Results */}
      {hasData && (
        <div className="mt-10 space-y-6">
          {/* File info card */}
          <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <FileSpreadsheet className="h-5 w-5 text-blue-500" />
                <h2 className="text-lg font-semibold text-slate-900 dark:text-white">
                  {filename}
                </h2>
              </div>
              <button
                onClick={() => router.push("/explore")}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 transition-colors"
              >
                Start Exploring
                <ArrowRight className="h-4 w-4" />
              </button>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
              <div className="flex items-center gap-2 text-sm">
                <Rows3 className="h-4 w-4 text-slate-400" />
                <span className="text-slate-500 dark:text-slate-400">Rows:</span>
                <span className="font-medium text-slate-900 dark:text-white">
                  {shape?.[0]?.toLocaleString()}
                </span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <Columns3 className="h-4 w-4 text-slate-400" />
                <span className="text-slate-500 dark:text-slate-400">Columns:</span>
                <span className="font-medium text-slate-900 dark:text-white">
                  {shape?.[1]}
                </span>
              </div>
            </div>

            {/* Column badges */}
            <div className="flex flex-wrap gap-1.5 mb-6">
              {columns.map((col: ColumnInfo) => (
                <span
                  key={col.name}
                  className={`
                    inline-flex items-center px-2 py-0.5 rounded text-xs font-medium
                    ${
                      col.semantic_type === "numeric"
                        ? "bg-emerald-50 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300"
                        : col.semantic_type === "categorical"
                        ? "bg-purple-50 text-purple-700 dark:bg-purple-950 dark:text-purple-300"
                        : col.semantic_type === "datetime"
                        ? "bg-amber-50 text-amber-700 dark:bg-amber-950 dark:text-amber-300"
                        : col.semantic_type === "boolean"
                        ? "bg-sky-50 text-sky-700 dark:bg-sky-950 dark:text-sky-300"
                        : "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300"
                    }
                  `}
                  title={`${col.dtype} | ${col.n_unique} unique | ${col.null_pct}% null`}
                >
                  {col.name}
                </span>
              ))}
            </div>
          </div>

          {/* Preview table */}
          <div>
            <h3 className="text-sm font-medium text-slate-500 dark:text-slate-400 mb-2">
              Data Preview
            </h3>
            <DataTable
              columns={columns.map((c: ColumnInfo) => c.name)}
              data={preview}
              maxRows={20}
            />
          </div>
        </div>
      )}
    </div>
  );
}
