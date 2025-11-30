# cURL Examples for RAG POC API

## 1. Load PDF Endpoint

### Basic cURL command:
```bash
curl -X POST "http://localhost:8000/load-pdf" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@resources/Questionnaire_for_Improving_Liquidity_in_Bond_Market.pdf"
```

### Using a different PDF file:
```bash
curl -X POST "http://localhost:8000/load-pdf" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@resources/nke-10k-2023.pdf"
```

### With verbose output to see response:
```bash
curl -X POST "http://localhost:8000/load-pdf" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@resources/Questionnaire_for_Improving_Liquidity_in_Bond_Market.pdf" \
  -v
```

### Pretty formatted JSON response:
```bash
curl -X POST "http://localhost:8000/load-pdf" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@resources/Questionnaire_for_Improving_Liquidity_in_Bond_Market.pdf" \
  | python -m json.tool
```

---

## 2. Query Endpoint

### Basic query:
```bash
curl -X POST "http://localhost:8000/query" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"What technologies are used in this project?\"}"
```

### Another example query:
```bash
curl -X POST "http://localhost:8000/query" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"Explain the main concepts discussed in the document\"}"
```

### Using a file for the JSON payload:
```bash
# Create query.json file:
echo '{"query": "What is the purpose of this document?"}' > query.json

# Then use it:
curl -X POST "http://localhost:8000/query" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d @query.json
```

### Pretty formatted JSON response:
```bash
curl -X POST "http://localhost:8000/query" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"What technologies are used in this project?\"}" \
  | python -m json.tool
```

---

## 3. Additional Endpoints

### Health Check:
```bash
curl -X GET "http://localhost:8000/health" \
  -H "accept: application/json"
```

### Root endpoint (API info):
```bash
curl -X GET "http://localhost:8000/" \
  -H "accept: application/json"
```

---

## Windows PowerShell Examples

### Load PDF (PowerShell):
```powershell
$filePath = "resources\Questionnaire_for_Improving_Liquidity_in_Bond_Market.pdf"
$uri = "http://localhost:8000/load-pdf"

$form = @{
    file = Get-Item -Path $filePath
}

Invoke-RestMethod -Uri $uri -Method Post -Form $form
```

### Query (PowerShell):
```powershell
$uri = "http://localhost:8000/query"
$body = @{
    query = "What technologies are used in this project?"
} | ConvertTo-Json

Invoke-RestMethod -Uri $uri -Method Post -Body $body -ContentType "application/json"
```

---

## Complete Workflow Example

```bash
# Step 1: Load a PDF
curl -X POST "http://localhost:8000/load-pdf" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@resources/Questionnaire_for_Improving_Liquidity_in_Bond_Market.pdf"

# Step 2: Query the loaded document
curl -X POST "http://localhost:8000/query" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"What technologies are used in this project?\"}"
```




