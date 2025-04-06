from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import os
import re
import pandas as pd
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from werkzeug.utils import secure_filename
import uuid
import json
from datetime import datetime
import time

# Configure Flask app
app = Flask(__name__)
app.secret_key = 'bank_statement_analyzer_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'csv', 'xlsx', 'xls'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# from google.cloud import vision
# import io

# def extract_text_from_pdf(pdf_path):
#     images = convert_from_path(pdf_path)
#     all_text = ""
    
#     # Initialize Vision client
#     client = vision.ImageAnnotatorClient()
    
#     for image in images:
#         # Convert PIL image to bytes
#         img_byte_arr = io.BytesIO()
#         image.save(img_byte_arr, format='PNG')
#         content = img_byte_arr.getvalue()
        
#         # Create image object
#         vision_image = vision.Image(content=content)
        
#         # Perform OCR
#         response = client.text_detection(image=vision_image)
#         texts = response.text_annotations
        
#         if texts:
#             all_text += texts[0].description + "\n"
    
#     return all_text

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using OCR"""
    images = convert_from_path(pdf_path)
    all_text = ""
    
    for image in images:
        # Convert to OpenCV format
        open_cv_image = np.array(image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
        
        # Preprocess image for better OCR
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # OCR
        text = pytesseract.image_to_string(thresh)
        all_text += text + "\n"
        
    return all_text

def parse_csv_excel(file_path):
    """Parse CSV or Excel file"""
    if file_path.lower().endswith('.csv'):
        df = pd.read_csv(file_path)
    else:  # Excel formats
        df = pd.read_excel(file_path)
        
    return df

def process_pdf_statement(text):
    """Process the extracted text from a PDF bank statement"""
    lines = text.split('\n')
    transactions = []
    
    # Common patterns in bank statements
    date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
    amount_pattern = r'[-+]?\$?\s*[\d,]+\.\d{2}'
    
    for line in lines:
        # Skip headers and empty lines
        if not line.strip() or 'Date' in line and 'Description' in line:
            continue
            
        # Try to identify a transaction line
        if re.search(date_pattern, line):
            # Extract date
            date_match = re.search(date_pattern, line)
            date = date_match.group(0) if date_match else ""
            
            # Extract amount
            amount_matches = re.findall(amount_pattern, line)
            
            if amount_matches:
                # Determine if it's debit or credit
                description = line
                
                for amount_str in amount_matches:
                    # Clean the amount string
                    amount_str = amount_str.replace('$', '').replace(',', '').strip()
                    amount = float(amount_str)
                    
                    # Determine if debit or credit
                    if '-' in amount_str:
                        transactions.append({
                            'type': 'debit',
                            'amount': abs(amount),
                            'date': date,
                            'description': description
                        })
                    else:
                        transactions.append({
                            'type': 'credit',
                            'amount': amount,
                            'date': date,
                            'description': description
                        })
    
    return transactions

def analyze_statement(file_path):
    """Process any supported bank statement format and calculate totals"""
    result = {
        'total_debits': 0,
        'total_credits': 0,
        'transactions': [],
        'monthly_summary': {},
        'category_summary': {}
    }
    
    if file_path.lower().endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
        transactions = process_pdf_statement(text)
        
        # Calculate totals
        result['total_debits'] = sum(t['amount'] for t in transactions if t['type'] == 'debit')
        result['total_credits'] = sum(t['amount'] for t in transactions if t['type'] == 'credit')
        result['transactions'] = transactions
        
        # Create DataFrame for additional analysis
        if transactions:
            df = pd.DataFrame(transactions)
        else:
            print(result)
            return result
    
    elif file_path.lower().endswith(('.csv', '.xlsx', '.xls')):
        df = parse_csv_excel(file_path)
        
        # Try to identify columns
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['date', 'day']):
                column_mapping['date'] = col
            elif any(term in col_lower for term in ['desc', 'narration', 'particular', 'detail']):
                column_mapping['description'] = col
            elif any(term in col_lower for term in ['debit', 'withdrawal', 'payment', 'sent']):
                column_mapping['debit'] = col
            elif any(term in col_lower for term in ['credit', 'deposit', 'received']):
                column_mapping['credit'] = col
            elif any(term in col_lower for term in ['amount', 'transaction']):
                column_mapping['amount'] = col
            elif any(term in col_lower for term in ['balance']):
                column_mapping['balance'] = col
            elif any(term in col_lower for term in ['category', 'type']):
                column_mapping['category'] = col
        
        # Process based on identified columns
        transactions = []
        
        # If we have separate debit and credit columns
        if 'debit' in column_mapping and 'credit' in column_mapping:
            # Convert to numeric
            df[column_mapping['debit']] = pd.to_numeric(df[column_mapping['debit']].astype(str).str.replace('[$,]', '', regex=True), errors='coerce').fillna(0)
            df[column_mapping['credit']] = pd.to_numeric(df[column_mapping['credit']].astype(str).str.replace('[$,]', '', regex=True), errors='coerce').fillna(0)
            
            for _, row in df.iterrows():
                debit_amount = row[column_mapping['debit']]
                credit_amount = row[column_mapping['credit']]
                
                # Only add non-zero transactions
                if debit_amount > 0:
                    transactions.append({
                        'type': 'debit',
                        'amount': debit_amount,
                        'date': row.get(column_mapping.get('date', ''), ''),
                        'description': row.get(column_mapping.get('description', ''), ''),
                        'category': row.get(column_mapping.get('category', ''), 'Uncategorized')
                    })
                
                if credit_amount > 0:
                    transactions.append({
                        'type': 'credit',
                        'amount': credit_amount,
                        'date': row.get(column_mapping.get('date', ''), ''),
                        'description': row.get(column_mapping.get('description', ''), ''),
                        'category': row.get(column_mapping.get('category', ''), 'Uncategorized')
                    })
        
        # If we have a single amount column
        elif 'amount' in column_mapping:
            # Convert to numeric
            df[column_mapping['amount']] = pd.to_numeric(df[column_mapping['amount']].astype(str).str.replace('[$,]', '', regex=True), errors='coerce')
            
            for _, row in df.iterrows():
                amount = row[column_mapping['amount']]
                if pd.isna(amount):
                    continue
                    
                if amount < 0:
                    transactions.append({
                        'type': 'debit',
                        'amount': abs(amount),
                        'date': row.get(column_mapping.get('date', ''), ''),
                        'description': row.get(column_mapping.get('description', ''), ''),
                        'category': row.get(column_mapping.get('category', ''), 'Uncategorized')
                    })
                else:
                    transactions.append({
                        'type': 'credit',
                        'amount': amount,
                        'date': row.get(column_mapping.get('date', ''), ''),
                        'description': row.get(column_mapping.get('description', ''), ''),
                        'category': row.get(column_mapping.get('category', ''), 'Uncategorized')
                    })
        
        # Calculate totals
        result['total_debits'] = sum(t['amount'] for t in transactions if t['type'] == 'debit')
        result['total_credits'] = sum(t['amount'] for t in transactions if t['type'] == 'credit')
        result['transactions'] = transactions
    
    else:
        return result
    
    # If we have transactions with dates, create monthly summary
    df_transactions = pd.DataFrame(result['transactions'])
    if not df_transactions.empty and 'date' in df_transactions.columns:
        try:
            # Try to convert dates to datetime format
            df_transactions['date'] = pd.to_datetime(df_transactions['date'], errors='coerce')
            df_transactions['month'] = df_transactions['date'].dt.strftime('%Y-%m')
            
            # Group by month
            monthly_data = df_transactions.groupby(['month', 'type'])['amount'].sum().unstack(fill_value=0).reset_index()
            
            # Convert to dictionary for JSON
            for _, row in monthly_data.iterrows():
                month = row['month']
                result['monthly_summary'][month] = {
                    'debit': float(row.get('debit', 0)),
                    'credit': float(row.get('credit', 0)),
                    'net': float(row.get('credit', 0) - row.get('debit', 0))
                }
        except:
            # If date conversion fails, skip monthly summary
            pass
    
    # Create category summary if available
    if not df_transactions.empty and 'category' in df_transactions.columns:
        category_data = df_transactions.groupby(['category', 'type'])['amount'].sum().unstack(fill_value=0).reset_index()
        
        for _, row in category_data.iterrows():
            category = row['category']
            result['category_summary'][category] = {
                'debit': float(row.get('debit', 0)),
                'credit': float(row.get('credit', 0)),
                'net': float(row.get('credit', 0) - row.get('debit', 0))
            }
    
    return result

def generate_plots(analysis_data):
    """Generate visualizations for the dashboard"""
    plots = {}
    
    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = ['Debits', 'Credits', 'Net']
    values = [
        analysis_data['total_debits'], 
        analysis_data['total_credits'], 
        analysis_data['total_credits'] - analysis_data['total_debits']
    ]
    colors = ['#FF6B6B', '#4ECB71', '#3D7DD8']
    
    bars = ax.bar(labels, values, color=colors)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${abs(height):.2f}',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_title('Statement Summary', fontsize=14)
    ax.set_ylabel('Amount ($)', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    plots['summary'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    
    # Monthly trends if available
    if analysis_data['monthly_summary']:
        fig, ax = plt.subplots(figsize=(12, 6))
        months = list(analysis_data['monthly_summary'].keys())
        debits = [analysis_data['monthly_summary'][m]['debit'] for m in months]
        credits = [analysis_data['monthly_summary'][m]['credit'] for m in months]
        net = [analysis_data['monthly_summary'][m]['net'] for m in months]
        
        ax.plot(months, credits, 'g-', marker='o', label='Credits')
        ax.plot(months, debits, 'r-', marker='o', label='Debits')
        ax.plot(months, net, 'b-', marker='s', label='Net')
        
        ax.set_title('Monthly Trends', fontsize=14)
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Amount ($)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        if len(months) > 6:
            plt.xticks(rotation=45)
        
        # Save plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        plots['monthly'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
    
    # Category breakdown if available
    if analysis_data['category_summary']:
        fig, ax = plt.subplots(figsize=(10, 8))
        categories = list(analysis_data['category_summary'].keys())
        debits = [analysis_data['category_summary'][c]['debit'] for c in categories]
        
        # Sort for better visualization
        sorted_data = sorted(zip(categories, debits), key=lambda x: x[1], reverse=True)
        categories = [item[0] for item in sorted_data]
        debits = [item[1] for item in sorted_data]
        
        # Only show top 10 categories if there are many
        if len(categories) > 10:
            categories = categories[:10]
            debits = debits[:10]
            ax.set_title('Top 10 Expense Categories', fontsize=14)
        else:
            ax.set_title('Expense Categories', fontsize=14)
        
        bars = ax.barh(categories, debits, color='#FF6B6B')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                    f'${width:.2f}',
                    ha='left', va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Amount ($)', fontsize=12)
        ax.set_ylabel('Category', fontsize=12)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Save plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        plots['category'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
    
    return plots

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Generate unique ID for this analysis
        analysis_id = str(uuid.uuid4())
        
        # Create directory for this analysis
        analysis_dir = os.path.join(app.config['UPLOAD_FOLDER'], analysis_id)
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(analysis_dir, filename)
        file.save(file_path)
        
        # Process the file
        try:
            analysis_data = analyze_statement(file_path)
            
            # Generate plots
            plots = generate_plots(analysis_data)
            
            # Save analysis results
            analysis_data['file_name'] = filename
            analysis_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            analysis_data['plots'] = plots
            
            # Save to JSON file
            with open(os.path.join(analysis_dir, 'analysis.json'), 'w') as f:
                json.dump(analysis_data, f)
            
            # Store analysis ID in session
            session['current_analysis'] = analysis_id
            
            return redirect(url_for('dashboard', analysis_id=analysis_id))
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('index'))
    
    flash('File type not allowed')
    return redirect(url_for('index'))

@app.route('/dashboard/<analysis_id>')
def dashboard(analysis_id):
    # Check if analysis exists
    analysis_dir = os.path.join(app.config['UPLOAD_FOLDER'], analysis_id)
    analysis_file = os.path.join(analysis_dir, 'analysis.json')
    
    if not os.path.exists(analysis_file):
        flash('Analysis not found')
        return redirect(url_for('index'))
    
    # Load analysis data
    with open(analysis_file, 'r') as f:
        analysis_data = json.load(f)
    
    return render_template('dashboard.html', 
                          analysis=analysis_data,
                          analysis_id=analysis_id)

@app.route('/api/transactions/<analysis_id>')
def get_transactions(analysis_id):
    # Check if analysis exists
    analysis_dir = os.path.join(app.config['UPLOAD_FOLDER'], analysis_id)
    analysis_file = os.path.join(analysis_dir, 'analysis.json')
    
    if not os.path.exists(analysis_file):
        return jsonify({'error': 'Analysis not found'}), 404
    
    # Load analysis data
    with open(analysis_file, 'r') as f:
        analysis_data = json.load(f)
    
    # Pagination parameters
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    search = request.args.get('search', '')
    
    transactions = analysis_data['transactions']
    
    # Apply search filter if provided
    if search:
        search_lower = search.lower()
        transactions = [t for t in transactions if search_lower in str(t.get('description', '')).lower()]
    
    # Calculate pagination
    total = len(transactions)
    total_pages = (total + per_page - 1) // per_page
    
    # Validate page number
    if page < 1:
        page = 1
    if page > total_pages and total_pages > 0:
        page = total_pages
    
    # Get transactions for current page
    start = (page - 1) * per_page
    end = start + per_page
    current_transactions = transactions[start:end]
    
    return jsonify({
        'transactions': current_transactions,
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total': total,
            'total_pages': total_pages
        }
    })

@app.route('/history')
def history():
    """View analysis history"""
    analyses = []
    
    # List all directories in the upload folder
    for analysis_id in os.listdir(app.config['UPLOAD_FOLDER']):
        analysis_dir = os.path.join(app.config['UPLOAD_FOLDER'], analysis_id)
        analysis_file = os.path.join(analysis_dir, 'analysis.json')
        
        if os.path.exists(analysis_file):
            with open(analysis_file, 'r') as f:
                analysis_data = json.load(f)
                
                analyses.append({
                    'id': analysis_id,
                    'file_name': analysis_data.get('file_name', 'Unknown'),
                    'timestamp': analysis_data.get('timestamp', ''),
                    'total_debits': analysis_data.get('total_debits', 0),
                    'total_credits': analysis_data.get('total_credits', 0),
                    'net': analysis_data.get('total_credits', 0) - analysis_data.get('total_debits', 0)
                })
    
    # Sort by timestamp (newest first)
    analyses.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return render_template('history.html', analyses=analyses)

if __name__ == '__main__':
    app.run(debug=True)