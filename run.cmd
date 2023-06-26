echo "Starting the application..."
start cmd.exe /C "cd client && npm run dev"
start cmd.exe /C "cd server && npm start"
start cmd.exe /C ".\venv\Scripts\activate && mingw32-make start"