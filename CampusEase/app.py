# FILE: app.py
import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from model.ml_model import extract_features, find_most_similar_image
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin, current_user


from functools import wraps
from flask import abort

from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# -------------------- APP CONFIG --------------------
app = Flask(__name__)
app.secret_key = "supersecretkey"  # Change this in production

# 🔹 MySQL (XAMPP) Connection
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/campusease'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 🔹 Upload Folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))



# -------------------- DATABASE MODELS --------------------



class User(db.Model, UserMixin):
    __tablename__ = 'user'   # ✅ keep this since your table is named 'user'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    gender = db.Column(db.String(10))
    dob = db.Column(db.Date)
    phone = db.Column(db.String(20))
    profile_pic = db.Column(db.String(255))





class LostFoundItem(db.Model):
    __tablename__ = 'lost_found_items'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    description = db.Column(db.Text)
    status = db.Column(db.Enum('Lost', 'Found'), nullable=False)
    owner_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    contact = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(100))
    category = db.Column(db.Enum('Electronics', 'Clothing', 'Documents', 'Accessories', 'Bag',))
    image_path = db.Column(db.String(200))
    image_features = db.Column(db.LargeBinary)
    matched_item_id = db.Column(db.Integer, db.ForeignKey('lost_found_items.id'))
    date_reported = db.Column(db.DateTime, default=db.func.current_timestamp())

    owner = db.relationship('User', backref='lost_found_items')
    

from datetime import datetime
# or from your main file where db = SQLAlchemy(app)

class Preorder(db.Model):
    __tablename__ = 'preorders'
    __table_args__ = {'extend_existing': True}


    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE', onupdate='CASCADE'), nullable=False)
    item_name = db.Column(db.String(150), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    total_price = db.Column(db.Float, nullable=False)
    order_status = db.Column(db.Enum('pending', 'ready', 'delivered', name='order_status_enum'), default='pending')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationship back to user
    user = db.relationship('User', backref=db.backref('preorders', cascade='all, delete-orphan'))

    def __repr__(self):
        return f"<Preorder {self.item_name} by User {self.user_id}>"




# ✅ Folder to store uploaded profile pictures
app.config['UPLOAD_FOLDER'] = 'static/profile_pics'

# -------------------- LOAD ML MODEL --------------------
try:
    model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
except Exception as e:
    print(f"⚠️ Error loading ML model: {e}")
    model = None


def extract_features(image_path):
    """Extract deep features from image using MobileNetV2."""
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array)
        return features.flatten()
    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        return None


# -------------------- ROUTES --------------------
@app.route('/')
def index():
    return render_template('index.html')

# Pre-order Page (placeholder for now)
@app.route('/user_preorder')
def user_preorder():
    # ✅ Check if user is logged in
    if 'user_id' not in session or session.get('role') != 'user':
        flash("Please login to access this page.")
        return redirect(url_for('login'))

    user_id = session.get('user_id')
    user = User.query.get(user_id)

    # ✅ Fetch orders for the logged-in user
    user_orders = []
    try:
        orders = PreOrder.query.filter_by(user_id=user_id).all()
        for o in orders:
            user_orders.append({
                'id': o.id,
                'item_name': o.item_name,
                'quantity': o.quantity,
                'total_price': o.total_price,
                'date_ordered': o.date_ordered
            })
    except Exception as e:
        print(f"Error fetching orders: {e}")

    return render_template('user_preorder.html', user=user, orders=user_orders)


@app.route('/place_order', methods=['POST'])
def place_order():
    if 'user_id' not in session:
        flash("Please log in first.")
        return redirect(url_for('login'))

    item_name = request.form.get('item_name')
    price = float(request.form.get('price'))
    quantity = int(request.form.get('quantity'))
    total_price = price * quantity
    user_id = session['user_id']

    new_order = Preorder(
        user_id=user_id,
        item_name=item_name,
        quantity=quantity,
        total_price=total_price
    )
    db.session.add(new_order)
    db.session.commit()

    flash(f"✅ You have successfully pre-ordered {quantity} × {item_name}!")
    return redirect(url_for('user_preorder'))



def role_required(role):
    """Restrict access to users with the given role."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not current_user.is_authenticated:
                return redirect(url_for('login'))
            if current_user.role != role:
                abort(403)  # Forbidden
            return func(*args, **kwargs)
        return wrapper
    return decorator





@app.route('/admin_preorders')
def admin_preorders():
    if session.get('role') != 'admin':
        flash("Access denied.")
        return redirect(url_for('login'))

    orders = Preorder.query.order_by(Preorder.date_ordered.desc()).all()
    return render_template('admin_preorders.html', orders=orders)

@app.route('/update_order/<int:order_id>/<string:new_status>')
def update_order(order_id, new_status):
    if session.get('role') != 'admin':
        flash("Access denied.")
        return redirect(url_for('login'))

    order = Preorder.query.get(order_id)
    if order:
        order.status = new_status
        db.session.commit()
        flash(f"Order #{order.id} marked as {new_status}.")
    else:
        flash("Order not found.")
    
    return redirect(url_for('admin_preorders'))


# ---------- Register ----------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        role = request.form['role']

        if password != confirm_password:
            flash("Passwords do not match.")
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash("Email already registered.")
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        new_user = User(name=name, email=email, password=hashed_password, role=role)
        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful! Please log in.")
        return redirect(url_for('login'))

    return render_template('register.html')



# ---------- Login ----------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role')  # user/admin/canteen

        # ✅ Step 1: Basic validation
        if not email or not password or not role:
            flash("Please fill all fields.")
            return redirect(url_for('login'))

        # ✅ Step 2: Find the user based on email
        user = User.query.filter_by(email=email).first()

        if not user:
            flash("User not found.")
            return redirect(url_for('login'))

        # ✅ Step 3: Verify password and role
        if not check_password_hash(user.password, password):
            flash("Invalid password.")
            return redirect(url_for('login'))

        if user.role != role:
            flash(f"You are registered as a {user.role}, not as {role}.")
            return redirect(url_for('login'))

        # ✅ Step 4: Set session variables
        session['user_id'] = user.id
        session['user_name'] = user.name
        session['role'] = user.role

        flash("Login successful!")

        # ✅ Step 5: Redirect based on role
        if user.role == 'admin':
            return redirect(url_for('admin_dashboard'))
        elif user.role == 'canteen':
            return redirect(url_for('canteen_dashboard'))
        else:
            return redirect(url_for('user_dashboard'))

    return render_template('login.html')




# ---------- Lost & Found ----------
@app.route('/lost_found', methods=['GET', 'POST'])
def lost_found():
    if 'user_id' not in session:
        flash("Please log in first.")
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            name = request.form['name']
            description = request.form['description']
            status = request.form['status']
            contact = request.form['contact']
            location = request.form['location']
            category = request.form.get('category') or "Other"
            image = request.files['image']

            print(f"🟦 Received item: {name}, {status}, {contact}, {location}, {category}")

            # --- Save image properly ---
            image_filename = None
            features_bytes = None

            if image and image.filename != '':
                image_filename = secure_filename(image.filename)
                save_path = os.path.join('static', 'img', image_filename)
                image.save(save_path)
                print(f"✅ Image saved to: {save_path}")

                # Extract ML features if model is loaded
                if model:
                    features = extract_features(save_path)
                    if features is not None:
                        features_bytes = pickle.dumps(features)
                        print("✅ Features extracted successfully.")
                    else:
                        print("⚠️ Failed to extract features.")
                else:
                    print("⚠️ ML model not loaded, skipping feature extraction.")

            # --- Add new item to DB ---
            new_item = LostFoundItem(
                name=name,
                description=description,
                status=status,
                owner_id=session['user_id'],
                contact=contact,
                location=location,
                category=category,
                image_path=image_filename,
                image_features=features_bytes
            )

            db.session.add(new_item)
            db.session.commit()
            print(f"✅ Item '{name}' added to database.")

            # --- Flash success message ---
            flash("Your item has been posted successfully!")
            return redirect(url_for('lost_found'))

        except Exception as e:
            db.session.rollback()
            print(f"❌ Error adding item: {e}")
            flash("Something went wrong while posting the item.")
            return redirect(url_for('lost_found'))

    # --- GET request: show all items ---
    items = LostFoundItem.query.order_by(LostFoundItem.date_reported.desc()).all()
    for item in items:
        owner = User.query.get(item.owner_id)
        item.owner_name = owner.name if owner else "Unknown"

    return render_template('lost_found.html', lost_found_items=items)


# ---------- Dashboards ----------


from sqlalchemy import func
from datetime import datetime, timedelta

from collections import defaultdict

@app.route('/admin_dashboard')
def admin_dashboard():
    if session.get('role') != 'admin':
        flash("Access denied.")
        return redirect(url_for('login'))

    # ✅ Summary statistics
    stats = {
        "total_users": User.query.count(),
        "lost_items": LostFoundItem.query.filter_by(status='Lost').count(),
        "found_items": LostFoundItem.query.filter_by(status='Found').count(),
        "matched_items": LostFoundItem.query.filter(LostFoundItem.matched_item_id.isnot(None)).count()
    }

    # ✅ Fetch recent items
    recent_items = LostFoundItem.query.order_by(LostFoundItem.date_reported.desc()).limit(5).all()

    # ✅ ML/analytics prediction data
    ml_predictions = {
        "system_efficiency": 92,
        "most_reported_category": "Electronics",
        "weekly_lost_items": 18,
        "weekly_found_items": 12
    }

    # ✅ Prepare monthly trend data for chart
    monthly_data = defaultdict(lambda: {"Lost": 0, "Found": 0})
    all_items = LostFoundItem.query.all()

    for item in all_items:
        if item.date_reported:
            month = item.date_reported.strftime('%b')  # e.g., Jan, Feb
            if item.status in ['Lost', 'Found']:
                monthly_data[month][item.status] += 1

    # Sort months by calendar order
    ordered_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    months = [m for m in ordered_months if m in monthly_data]
    lost_counts = [monthly_data[m]["Lost"] for m in months]
    found_counts = [monthly_data[m]["Found"] for m in months]

    return render_template(
        'admin_dashboard.html',
        stats=stats,
        recent_items=recent_items,
        ml_predictions=ml_predictions,
        months=months,
        lost_counts=lost_counts,
        found_counts=found_counts
    )




@app.route('/user_dashboard')
def user_dashboard():
    # ✅ Check if user is logged in
    if 'user_id' not in session:
        flash("Please log in first.")
        return redirect(url_for('login'))

    user_id = session['user_id']
    user = User.query.get(user_id)

    if not user:
        flash("User not found.")
        return redirect(url_for('login'))

    # ✅ Fetch user-specific data safely
    stats = {
        "my_lost_items": LostFoundItem.query.filter_by(owner_id=user.id, status='Lost').count(),
        "my_found_items": LostFoundItem.query.filter_by(owner_id=user.id, status='Found').count(),
    }

    recent_activity = LostFoundItem.query.filter_by(owner_id=user.id)\
                        .order_by(LostFoundItem.date_reported.desc())\
                        .limit(5).all()

    return render_template('user_dashboard.html', user=user, stats=stats, recent_activity=recent_activity)





@app.route('/canteen_dashboard')
@login_required
@role_required('canteen')
def canteen_dashboard():
    # Fetch all pending or active orders for the canteen
    pending_orders = Preorder.query.filter_by(order_status='pending').all()
    ready_orders = Preorder.query.filter_by(order_status='ready').all()
    delivered_orders = Preorder.query.filter_by(order_status='delivered').all()

    return render_template('canteen_dashboard.html',
                           user=current_user,
                           pending_orders=pending_orders,
                           ready_orders=ready_orders,
                           delivered_orders=delivered_orders)





@app.errorhandler(403)
def forbidden(e):
    return render_template('403.html'), 403





# ---------- Profile ----------
@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        flash('Please log in first!')
        return redirect(url_for('login'))

    user_id = session['user_id']
    user = User.query.get(user_id)

    if request.method == 'POST':
        gender = request.form.get('gender')
        dob = request.form.get('dob')
        phone = request.form.get('phone')

        if 'profile_pic' in request.files and request.files['profile_pic'].filename != '':
            file = request.files['profile_pic']
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            user.profile_pic = f"profile_pics/{filename}"

        user.gender = gender
        user.dob = dob
        user.phone = phone

        db.session.commit()
        flash('✅ Profile updated successfully!')
        return redirect(url_for('profile'))

    return render_template('profile.html', user=user)

# ---------- Logout ----------
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("You have been logged out.")
    return redirect(url_for('login'))


from flask import request, render_template, flash, redirect, url_for, session
from model.ml_model import get_recommendations  # make sure ml_model.py exists

@app.route('/search', methods=['POST'])
def search():
    if 'user_id' not in session:
        flash("Please log in first.")
        return redirect(url_for('login'))

    uploaded_image = request.files.get('image_search')
    matched_item = None
    message = None

    if not uploaded_image or uploaded_image.filename == '':
        message = "Please upload an image to search."
        return render_template('search.html', matched_item=None, message=message)

    # Save uploaded image
    upload_folder = os.path.join('static', 'uploads')
    os.makedirs(upload_folder, exist_ok=True)
    upload_path = os.path.join(upload_folder, uploaded_image.filename)
    uploaded_image.save(upload_path)

    # Run ML search
    from model.ml_model import find_most_similar_image
    best_match_filename = find_most_similar_image(upload_path, folder_path='static/img')

    if best_match_filename:
        matched_item = LostFoundItem.query.filter_by(image_path=best_match_filename).first()
        if matched_item:
            message = f"Possible match found: {matched_item.name}"
        else:
            message = "No exact database record found, but a similar image exists."
    else:
        message = "❌ No similar items found. Try again with a clearer or closer image."

    return render_template('search.html', matched_item=matched_item, message=message)



@app.route('/manage_users')
def manage_users():
    if session.get('role') != 'admin':
        flash("Access denied.")
        return redirect(url_for('login'))
    users = User.query.all()
    return render_template('manage_users.html', users=users)

@app.route('/manage_lost_items')
def manage_lost_items():
    if session.get('role') != 'admin':
        flash("Access denied.")
        return redirect(url_for('login'))
    lost_items = LostFoundItem.query.filter_by(status='Lost').all()
    return render_template('manage_lost_items.html', items=lost_items)

@app.route('/manage_found_items')
def manage_found_items():
    if session.get('role') != 'admin':
        flash("Access denied.")
        return redirect(url_for('login'))
    found_items = LostFoundItem.query.filter_by(status='Found').all()
    return render_template('manage_found_items.html', items=found_items)


# -------------------- RUN APP --------------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
