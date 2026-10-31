// The backend sends the `_Serialize` shapes (every field present); these are the names the app uses.
export type {
  GpuCell_Serialize as GpuCell,
  GpuState_Serialize as GpuState,
  HostView_Serialize as HostView,
  LogChunk,
  QueueState_Serialize as QueueState,
  Task_Serialize as Task,
} from "./bindings";
