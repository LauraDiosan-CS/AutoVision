"use client";

import { useCallback, useState, type ChangeEvent, type DragEvent } from "react";
import { UploadDropzone } from "@/components/UploadDropzone";

interface AppHeaderProps {
  importJson: (file: File) => Promise<boolean>;
  activeFileName: string;
}

export function AppHeader({ importJson, activeFileName }: AppHeaderProps) {
  const [isDragActive, setIsDragActive] = useState(false);

  const handleFileSelection = useCallback(
    async (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;
      const wasImported = await importJson(file);
      if (!wasImported) {
        event.target.value = "";
        return;
      }
      event.target.value = "";
    },
    [importJson]
  );

  const handleDropZoneDragOver = useCallback(
    (event: DragEvent<HTMLLabelElement>) => {
      event.preventDefault();
      setIsDragActive(true);
    },
    []
  );

  const handleDropZoneDragLeave = useCallback(() => {
    setIsDragActive(false);
  }, []);

  const handleDropZoneFileDrop = useCallback(
    async (event: DragEvent<HTMLLabelElement>) => {
      event.preventDefault();
      setIsDragActive(false);
      const file = event.dataTransfer.files?.[0];
      if (!file) return;
      await importJson(file);
    },
    [importJson]
  );

  return (
    <header className="rounded-3xl border border-white/10 bg-white/[0.03] p-8 shadow-[0_20px_80px_-40px_rgba(15,118,110,0.75)] backdrop-blur">
      <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
        <div className="space-y-3">
          <p className="text-sm font-semibold uppercase tracking-[0.2em] text-cyan-300">
            Pipeline Composer
          </p>
          <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">
            Import, visualize, and refactor perception pipelines.
          </h1>
          <p className="text-sm text-slate-300">
            Load a JSON configuration to project each processing lane. Drag
            modules to reorder or re-route between lanes and instantly explore
            alternative topologies.
          </p>
          <div className="flex justify-center flex-row">
            <UploadDropzone
              isDragActive={isDragActive}
              activeFileName={activeFileName}
              onFileSelection={handleFileSelection}
              onDragOver={handleDropZoneDragOver}
              onDragEnter={handleDropZoneDragOver}
              onDragLeave={handleDropZoneDragLeave}
              onDrop={handleDropZoneFileDrop}
            />
          </div>
        </div>
      </div>
    </header>
  );
}
