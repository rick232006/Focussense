const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let pythonProcess;
let isDev = process.env.NODE_ENV === 'development';

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1600,
    height: 950,
    minWidth: 1200,
    minHeight: 700,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    backgroundColor: '#F5F7FA',
    title: 'FocusSense - Eye Tracking & Focus Analytics',
    icon: path.join(__dirname, 'assets', 'icon.png')
  });

  // Load the React app
  if (isDev) {
    mainWindow.loadURL('http://localhost:5173');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, 'dist', 'index.html'));
  }

  // Prevent navigation away from app
  mainWindow.webContents.on('will-navigate', (event, url) => {
    if (!url.startsWith('http://localhost') && !url.startsWith('file://')) {
      event.preventDefault();
    }
  });

  // Handle window close
  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

function startPythonBackend() {
  console.log('[Electron] Starting Python backend...');
  
  const pythonPath = isDev 
    ? 'python' 
    : path.join(process.resourcesPath, 'python', 'python.exe');
  
  const scriptPath = path.join(__dirname, 'backend.py');
  
  pythonProcess = spawn(pythonPath, [scriptPath]);
  
  pythonProcess.stdout.on('data', (data) => {
    console.log(`[Python] ${data}`);
  });
  
  pythonProcess.stderr.on('data', (data) => {
    console.error(`[Python Error] ${data}`);
  });
  
  pythonProcess.on('close', (code) => {
    console.log(`[Python] Exited with code ${code}`);
  });
  
  console.log('[Electron] Python backend started');
}

function stopPythonBackend() {
  if (pythonProcess) {
    console.log('[Electron] Stopping Python backend...');
    pythonProcess.kill();
    pythonProcess = null;
  }
}

// App lifecycle
app.whenReady().then(() => {
  createWindow();
  startPythonBackend();
  
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  stopPythonBackend();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('will-quit', () => {
  stopPythonBackend();
});

// IPC handlers
ipcMain.handle('get-backend-status', async () => {
  return {
    running: pythonProcess !== null,
    port: 8765
  };
});

ipcMain.handle('restart-backend', async () => {
  stopPythonBackend();
  setTimeout(() => {
    startPythonBackend();
  }, 1000);
  return { success: true };
});
