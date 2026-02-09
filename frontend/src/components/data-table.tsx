"use client";

interface DataTableProps {
  columns: string[];
  data: Record<string, unknown>[];
  maxRows?: number;
}

export function DataTable({ columns, data, maxRows = 50 }: DataTableProps) {
  const rows = data.slice(0, maxRows);

  if (columns.length === 0) {
    return (
      <p className="text-sm text-slate-500 dark:text-slate-400">
        No data to display.
      </p>
    );
  }

  return (
    <div className="overflow-auto rounded-lg border border-slate-200 dark:border-slate-800">
      <table className="min-w-full text-sm">
        <thead>
          <tr className="bg-slate-50 dark:bg-slate-900">
            {columns.map((col) => (
              <th
                key={col}
                className="px-3 py-2 text-left font-medium text-slate-600 dark:text-slate-300 whitespace-nowrap border-b border-slate-200 dark:border-slate-800"
              >
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={i}
              className="border-b border-slate-100 dark:border-slate-800/50 hover:bg-slate-50 dark:hover:bg-slate-900/50 transition-colors"
            >
              {columns.map((col) => (
                <td
                  key={col}
                  className="px-3 py-1.5 whitespace-nowrap text-slate-700 dark:text-slate-300 max-w-xs truncate"
                  title={String(row[col] ?? "")}
                >
                  {row[col] == null ? (
                    <span className="text-slate-300 dark:text-slate-600">
                      null
                    </span>
                  ) : (
                    String(row[col])
                  )}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {data.length > maxRows && (
        <p className="px-3 py-2 text-xs text-slate-400 dark:text-slate-500 bg-slate-50 dark:bg-slate-900">
          Showing {maxRows} of {data.length} rows
        </p>
      )}
    </div>
  );
}
