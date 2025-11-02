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
};

export function PipelineCanvas({
  pipelines,
  onAllowDrop,
  onModuleDrop,
  onCreateModule,
  onEditModule,
  onDragStart,
  onDragEnd,
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
            <div className="flex min-h-[280px] items-center justify-center text-sm text-slate-500">
              Import a pipeline json file to begin composing.
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
                  />
                ))}
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
