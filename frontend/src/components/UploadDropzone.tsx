import type { DragEvent, ChangeEventHandler } from "react";

type UploadDropzoneProps = {
  isDragActive: boolean;
  activeFileName: string;
  onFileSelection: ChangeEventHandler<HTMLInputElement>;
  onDragOver: (event: DragEvent<HTMLLabelElement>) => void;
  onDragEnter: (event: DragEvent<HTMLLabelElement>) => void;
  onDragLeave: (event: DragEvent<HTMLLabelElement>) => void;
  onDrop: (event: DragEvent<HTMLLabelElement>) => void;
};

export function UploadDropzone({
  isDragActive,
  activeFileName,
  onFileSelection,
  onDragOver,
  onDragEnter,
  onDragLeave,
  onDrop,
}: UploadDropzoneProps) {
  return (
    <div className="mt-8 flex flex-col gap-4 md:flex-row md:items-center md:justify-center">
      <label
        onDragOver={onDragOver}
        onDragEnter={onDragEnter}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        className={`group relative flex cursor-pointer flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed px-4 py-6 transition ${
          isDragActive
            ? "border-cyan-400 bg-cyan-400/10"
            : "border-white/10 bg-white/5 hover:border-cyan-400/60"
        }`}
      >
        <input type="file" accept="application/json" onChange={onFileSelection} className="hidden" />
        <div className="flex size-12 items-center justify-center rounded-full bg-cyan-400/10 text-cyan-300">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="size-6">
            <path d="M12 3a1 1 0 0 1 1 1v9.586l2.293-2.293a1 1 0 1 1 1.414 1.414l-4 4a1 1 0 0 1-1.414 0l-4-4a1 1 0 1 1 1.414-1.414L11 13.586V4a1 1 0 0 1 1-1Z" />
            <path d="M5 15a1 1 0 0 1 2 0v3h10v-3a1 1 0 1 1 2 0v4a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1v-4Z" />
          </svg>
        </div>
        <div className="text-center">
          <p className="font-medium text-slate-100">Drop a JSON file here or click to browse</p>
        </div>
        {activeFileName && (
          <span className="rounded-full bg-cyan-400/10 px-3 py-1 text-xs text-cyan-200">{activeFileName}</span>
        )}
      </label>
    </div>
  );
}
