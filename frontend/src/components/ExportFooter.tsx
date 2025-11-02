type ExportFooterProps = {
  onExport: () => void;
  disabled: boolean;
};

export function ExportFooter({ onExport, disabled }: ExportFooterProps) {
  return (
    <footer className="mt-10">
      <div className="rounded-3xl border border-white/10 bg-white/[0.03] p-6 shadow-[0_20px_60px_-40px_rgba(56,189,248,0.65)] backdrop-blur">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div className="space-y-1 text-sm text-slate-300">
            <p className="text-xs font-semibold uppercase tracking-[0.25em] text-cyan-300">Export json</p>
            <p>Download the current pipeline as a JSON file.</p>
          </div>
          <button
            type="button"
            onClick={onExport}
            disabled={disabled}
            className={`inline-flex items-center justify-center rounded-xl border px-5 py-3 text-sm font-semibold transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-300 ${
              disabled
                ? "cursor-not-allowed border-white/10 bg-white/5 text-slate-500"
                : "border-cyan-400/40 bg-cyan-500/20 text-cyan-100 hover:border-cyan-300/60 hover:bg-cyan-500/30"
            }`}
          >
            Export JSON
          </button>
        </div>
      </div>
    </footer>
  );
}
