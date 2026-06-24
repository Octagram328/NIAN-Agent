import { app, BrowserWindow, ipcMain } from 'electron';
import { spawn, ChildProcess } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';

let mainWindow: BrowserWindow | null = null;
let pythonProcess: ChildProcess | null = null;

/** 判断当前平台 */
const isMac = process.platform === 'darwin';
const isWin = process.platform === 'win32';

/**
 * 获取 Python 解释器路径。
 * 开发时使用项目根目录的 venv，打包后使用 resources/venv。
 */
function getPythonPath(): string {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, 'venv', 'Scripts', 'python.exe');
  }
  return path.join(__dirname, '..', 'venv', 'Scripts', 'python.exe');
}

/**
 * 获取项目根目录（Python 后端的工作目录）。
 * 开发时为项目根目录，打包后为 resources/app/（因为禁用了 asar）。
 */
function getProjectRoot(): string {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, 'app');
  }
  return path.join(__dirname, '..');
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 700,
    title: 'Nian Agent',
    // 无边框配置：Windows/Linux 用 frame: false；macOS 用 hidden titleBarStyle
    frame: false,
    titleBarStyle: isMac ? 'hidden' : 'default',
    transparent: false,
    backgroundColor: '#F8F5F0',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
    show: false,
  });

  // 显示加载页面
  const loadingPath = path.join(getProjectRoot(), 'web', 'static', 'loading.html');
  mainWindow.loadFile(loadingPath).catch(() => {
    // loading.html 不存在时直接等待后端
  });

  // 等待后端就绪后加载主界面
  waitForBackend(() => {
    mainWindow?.loadURL('http://127.0.0.1:8000');
    mainWindow?.show();
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // 监听最大化/还原事件，通知渲染进程更新按钮图标
  mainWindow.on('maximize', () => {
    mainWindow?.webContents.send('window-maximized', true);
  });
  mainWindow.on('unmaximize', () => {
    mainWindow?.webContents.send('window-maximized', false);
  });
}

function startBackend() {
  const projectRoot = getProjectRoot();
  const pythonPath = getPythonPath();

  console.log('[Electron] Starting Python backend...');
  console.log('[Electron] Python:', pythonPath);
  console.log('[Electron] CWD:', projectRoot);

  // 诊断：检查关键文件是否存在
  const envPath = path.join(projectRoot, '.env');
  const webServerPath = path.join(projectRoot, 'web', 'server.py');
  const srcPath = path.join(projectRoot, 'src');
  console.log('[Electron] .env exists:', fs.existsSync(envPath));
  console.log('[Electron] web/server.py exists:', fs.existsSync(webServerPath));
  console.log('[Electron] src/ exists:', fs.existsSync(srcPath));

  pythonProcess = spawn(
    pythonPath,
    ['-m', 'uvicorn', 'web.server:app', '--host', '127.0.0.1', '--port', '8000', '--log-level', 'error'],
    { cwd: projectRoot }
  );

  pythonProcess.stdout?.on('data', (data: Buffer) => {
    console.log(`[Python] ${data.toString().trim()}`);
  });

  pythonProcess.stderr?.on('data', (data: Buffer) => {
    console.error(`[Python] ${data.toString().trim()}`);
  });

  pythonProcess.on('exit', (code) => {
    console.log(`[Electron] Python backend exited with code ${code}`);
  });
}

function waitForBackend(callback: () => void) {
  const maxAttempts = 30;
  let attempts = 0;
  let isReady = false;
  const http = require('http');

  const check = () => {
    if (isReady) return;
    attempts++;
    const req = http.get('http://127.0.0.1:8000', (res: any) => {
      if (isReady) return;
      if (res.statusCode === 200) {
        isReady = true;
        console.log('[Electron] Backend is ready');
        callback();
      } else {
        retry();
      }
    });

    req.on('error', () => {
      if (!isReady) retry();
    });
    req.setTimeout(1000, () => {
      if (!isReady) {
        req.destroy();
        retry();
      }
    });
  };

  const retry = () => {
    if (isReady) return;
    if (attempts < maxAttempts) {
      setTimeout(check, 1000);
    } else {
      console.error('[Electron] Backend failed to start within 30 seconds');
      mainWindow?.show();
      mainWindow?.loadURL('data:text/html,<html><head><meta charset="utf-8"></head><body style="display:flex;align-items:center;justify-content:center;font-family:sans-serif;background:#F8F5F0;"><div style="text-align:center;color:#C23B22;"><h1>后端启动失败</h1><p>请检查 venv 和 Python 环境是否正确配置。</p></div></body></html>');
    }
  };

  check();
}

// ========== IPC 处理窗口控制事件 ==========

ipcMain.on('window-minimize', () => {
  mainWindow?.minimize();
});

ipcMain.on('window-maximize', () => {
  if (mainWindow?.isMaximized()) {
    mainWindow.unmaximize();
  } else {
    mainWindow?.maximize();
  }
});

ipcMain.on('window-close', () => {
  mainWindow?.close();
});

ipcMain.handle('window-is-maximized', () => {
  return mainWindow?.isMaximized() ?? false;
});

app.whenReady().then(() => {
  startBackend();
  createWindow();

  app.on('activate', () => {
    if (mainWindow === null) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (pythonProcess) {
    console.log('[Electron] Stopping Python backend...');
    pythonProcess.kill();
  }
  app.quit();
});
