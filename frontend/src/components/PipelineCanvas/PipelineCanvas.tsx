import type { DragEvent } from "react";
import type { PipelineLane as PipelineLaneType } from "@/types/pipeline";
import { PipelineLane } from "./PipelineLane";

type PipelineCanvasProps = {
  pipelines: PipelineLaneType[];
  onAllowDrop: (event: DragEvent<HTMLElement>) => void;
  onModuleDrop: (laneIndex: number, moduleIndex: number) => void;
  onCreateModule: (laneIndex: number, moduleIndex: number) => void;
  onEditModule: (laneIndex: number, moduleIndex: number) => void;
  onDragStart: (laneIndex: number, moduleIndex: number) => void;
  onDragEnd: () => void;
  onAddLane: () => void;
  onRenameLane: (laneIndex: number, label: string) => void;
  onDeleteLane: (laneIndex: number) => void;
};

export function PipelineCanvas({
  pipelines,
  onAllowDrop,
  onModuleDrop,
  onCreateModule,
  onEditModule,
  onDragStart,
  onDragEnd,
  onAddLane,
  onRenameLane,
  onDeleteLane,
}: PipelineCanvasProps) {
  return (
    <section className="flex flex-col gap-6">
      <div className="overflow-x-auto">
        <div className="inline-block min-w-full w-fit rounded-3xl border border-white/10 bg-slate-900/40 p-6 shadow-inner">
          <div className="space-y-3 mb-5">
            <p className="text-sm font-semibold uppercase tracking-[0.2em] text-cyan-300">
              Pipeline canvas
            </p>
            <p className="text-sm text-slate-300">
              Arrange and connect modules within each lane. Dragging a card into
              another lane reassigns its flow. Drop indicators appear as you
              hover over a lane or between cards.
            </p>
          </div>

          {pipelines.length === 0 ? (
            <div className="flex min-h-[280px] flex-col items-center justify-center gap-4 text-sm text-slate-400">
              <p className="max-w-sm text-center text-slate-400">
                Import a pipeline JSON file to begin composing, or create a
                lane to start from scratch.
              </p>
              <button
                type="button"
                onClick={onAddLane}
                className="flex items-center justify-center rounded-full border border-dashed border-cyan-400/40 bg-slate-950/60 px-5 py-2 text-sm font-semibold text-cyan-200 transition hover:border-cyan-300 hover:bg-cyan-500/10 hover:text-cyan-100 focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:ring-offset-2 focus:ring-offset-slate-900"
              >
                + Add lane
              </button>
            </div>
          ) : (
            <div className="grid gap-6 md:grid-cols-12">
              <div className="relative flex flex-col items-center justify-center gap-4 rounded-2xl border border-white/10 bg-white/5 px-6 py-10 text-center shadow-lg md:col-span-2">
                <span className="text-sm font-semibold uppercase tracking-[0.2em] text-cyan-300">
                  Source
                </span>
                <h3 className="text-lg font-semibold text-white">
                  Data Stream
                </h3>
                <p className="text-xs text-slate-400">
                  Origin input feeding every perception lane.
                </p>
              </div>

              <div className="flex flex-col gap-6 md:col-span-8">
                {pipelines.map((lane, laneIndex) => (
                  <PipelineLane
                    key={lane.id}
                    lane={lane}
                    laneIndex={laneIndex}
                    onAllowDrop={onAllowDrop}
                    onModuleDrop={onModuleDrop}
                    onCreateModule={onCreateModule}
                    onEditModule={onEditModule}
                    onDragStart={onDragStart}
                    onDragEnd={onDragEnd}
                    onRenameLane={onRenameLane}
                    onDeleteLane={onDeleteLane}
                  />
                ))}
                <button
                  type="button"
                  onClick={onAddLane}
                  className="flex h-14 items-center justify-center rounded-2xl border border-dashed border-cyan-400/40 bg-slate-950/40 text-sm font-semibold text-cyan-200 transition hover:border-cyan-300 hover:bg-cyan-500/10 hover:text-cyan-100 focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:ring-offset-2 focus:ring-offset-slate-900"
                >
                  + Add lane
                </button>
              </div>

              <div className="relative flex flex-col items-center justify-center gap-4 rounded-2xl border border-white/10 bg-white/5 px-6 py-10 text-center shadow-lg md:col-span-2">
                <span className="text-sm font-semibold uppercase tracking-[0.2em] text-blue-300">
                  Output
                </span>
                <h3 className="text-lg font-semibold text-white">
                  Data Merger
                </h3>
                <p className="text-xs text-slate-400">
                  Unified output after routing each perception stream.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
