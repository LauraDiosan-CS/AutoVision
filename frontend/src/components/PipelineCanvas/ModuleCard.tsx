import { formatValue } from "@/utils/pipeline";
import type { PipelineModule } from "@/types/pipeline";
import type { DragEvent, KeyboardEvent } from "react";

type ModuleCardProps = {
  moduleCard: PipelineModule;
  laneIndex: number;
  moduleIndex: number;
  onDragStart: (laneIndex: number, moduleIndex: number) => void;
  onDragEnd: () => void;
  onAllowDrop: (event: DragEvent<HTMLElement>) => void;
  onDrop: (laneIndex: number, moduleIndex: number) => void;
  onOpenEditor: (laneIndex: number, moduleIndex: number) => void;
};

export function ModuleCard({
  moduleCard,
  laneIndex,
  moduleIndex,
  onDragStart,
  onDragEnd,
  onAllowDrop,
  onDrop,
  onOpenEditor,
}: ModuleCardProps) {
  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onOpenEditor(laneIndex, moduleIndex);
    }
  };

  return (
    <div className="flex flex-col justify-center">
      <div
        role="button"
        tabIndex={0}
        draggable
        onDragStart={() => onDragStart(laneIndex, moduleIndex)}
        onDragEnd={onDragEnd}
        onDragOver={onAllowDrop}
        onDrop={(event) => {
          event.preventDefault();
          onDrop(laneIndex, moduleIndex);
        }}
        onClick={() => onOpenEditor(laneIndex, moduleIndex)}
        onKeyDown={handleKeyDown}
        className="min-w-[200px] rounded-2xl border bg-slate-950/60 p-3 text-left outline-none transition hover:-translate-y-1 hover:shadow-[0_20px_45px_-24px_rgba(56,189,248,0.65)] focus:-translate-y-1 focus:shadow-[0_20px_45px_-24px_rgba(56,189,248,0.65)]"
        style={{
          borderColor: `${moduleCard.accent}33`,
          boxShadow: `0 10px 30px -20px ${moduleCard.accent}80`,
        }}
      >
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="size-2 rounded-full" style={{ backgroundColor: moduleCard.accent }} />
            <p className="text-sm font-semibold text-white">{moduleCard.name}</p>
          </div>
          <span className="text-[10px] font-semibold uppercase tracking-widest text-slate-400">
            {`#${moduleIndex + 1}`}
          </span>
        </div>

        <div className="mt-3 flex flex-wrap gap-2">
          {Object.entries(moduleCard.config).map(([key, value]) => (
            <span key={key} className="rounded-full bg-white/5 px-3 py-1 text-[11px] text-slate-200">
              <span className="font-medium text-slate-300">{key}:</span> {formatValue(value)}
            </span>
          ))}
          {Object.keys(moduleCard.config).length === 0 && (
            <span className="rounded-full bg-white/5 px-3 py-1 text-[11px] text-slate-400">No attributes</span>
          )}
        </div>
      </div>
    </div>
  );
}
