import { contextBridge, ipcRenderer } from 'electron';

// 暴露安全的 API 给渲染进程
contextBridge.exposeInMainWorld('electronAPI', {
  platform: process.platform,

  // 窗口控制
  minimizeWindow: () => ipcRenderer.send('window-minimize'),
  maximizeWindow: () => ipcRenderer.send('window-maximize'),
  closeWindow: () => ipcRenderer.send('window-close'),
  isMaximized: () => ipcRenderer.invoke('window-is-maximized'),

  // 监听最大化状态变化（主进程 → 渲染进程）
  onMaximizeChange: (callback: (isMaximized: boolean) => void) => {
    const handler = (_: any, value: boolean) => callback(value);
    ipcRenderer.on('window-maximized', handler);
    // 返回解绑函数
    return () => ipcRenderer.removeListener('window-maximized', handler);
  },
});
