# 1️⃣ Use stable Python (NOT 3.13)
FROM python:3.10-slim

# 2️⃣ Set working directory inside container
WORKDIR /app

# 3️⃣ Copy requirements first (for faster builds)
COPY backend/requirements.txt .

# 4️⃣ Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5️⃣ Copy backend source code
COPY backend ./backend

# 6️⃣ Expose port Render uses
EXPOSE 10000

# 7️⃣ Start FastAPI app
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "10000"]
