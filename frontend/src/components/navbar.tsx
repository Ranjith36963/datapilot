"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BarChart3,
  Compass,
  FileText,
  MessageSquare,
  Moon,
  Sun,
  Upload,
} from "lucide-react";
import { useTheme } from "next-themes";
import { useSession } from "@/lib/store";

const navItems = [
  { href: "/", label: "Upload", icon: Upload },
  { href: "/explore", label: "Explore", icon: MessageSquare },
  { href: "/visualize", label: "Visualize", icon: BarChart3 },
  { href: "/export", label: "Export", icon: FileText },
];

export function Navbar() {
  const pathname = usePathname();
  const { theme, setTheme } = useTheme();
  const { filename, shape } = useSession();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  return (
    <nav className="border-b border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex h-14 items-center justify-between">
          {/* Logo */}
          <Link
            href="/"
            className="flex items-center gap-2 font-semibold text-lg text-slate-900 dark:text-white"
          >
            <Compass className="h-5 w-5 text-blue-600" />
            DataPilot
          </Link>

          {/* Navigation */}
          <div className="flex items-center gap-1">
            {navItems.map(({ href, label, icon: Icon }) => {
              const active = pathname === href;
              const disabled =
                href !== "/" && !filename;

              return (
                <Link
                  key={href}
                  href={disabled ? "#" : href}
                  className={`
                    flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium
                    transition-colors
                    ${
                      active
                        ? "bg-blue-50 text-blue-700 dark:bg-blue-950 dark:text-blue-300"
                        : disabled
                        ? "text-slate-300 dark:text-slate-700 cursor-not-allowed"
                        : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-50 dark:hover:bg-slate-900"
                    }
                  `}
                  onClick={disabled ? (e) => e.preventDefault() : undefined}
                >
                  <Icon className="h-4 w-4" />
                  {label}
                </Link>
              );
            })}
          </div>

          {/* Right side */}
          <div className="flex items-center gap-3">
            {filename && (
              <span className="text-xs text-slate-500 dark:text-slate-400 hidden sm:block">
                {filename}{" "}
                {shape && (
                  <span className="text-slate-400 dark:text-slate-600">
                    ({shape[0]}x{shape[1]})
                  </span>
                )}
              </span>
            )}
            <button
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              className="p-1.5 rounded-md text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
              aria-label="Toggle theme"
            >
              {mounted ? (
                theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />
              ) : (
                <div className="h-4 w-4" />
              )}
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
}
