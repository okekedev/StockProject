from dash import Dash, dcc, html, Input, Output, State, callback, no_update
import flask
import os
import dash_bootstrap_components as dbc
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user
from dotenv import load_dotenv
import config
from werkzeug.security import check_password_hash, generate_password_hash
# Before you set up routes or layouts
import callbacks.aiplus_import

# Load environment variables
load_dotenv()

# Get credentials from environment variables
VALID_USERNAME = os.getenv('DASH_USERNAME', 'admin')
VALID_PASSWORD_HASH = os.getenv('DASH_PASSWORD_HASH')

# If no password hash is set, create one for the default password 'password'
if not VALID_PASSWORD_HASH:
    VALID_PASSWORD_HASH = generate_password_hash('password')
    print(f"WARNING: No password hash found in environment. Using default password.")
    print(f"Set DASH_PASSWORD_HASH={VALID_PASSWORD_HASH} in your .env file for security.")

# Create a user class for Flask-Login
class User(UserMixin):
    def __init__(self, username):
        self.id = username

# Initialize Flask server
server = flask.Flask(__name__)
server.config.update(
    SECRET_KEY=os.getenv('SECRET_KEY', os.urandom(24).hex())
)

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(server)
login_manager.login_view = '/login'

# User loader callback
@login_manager.user_loader
def load_user(username):
    return User(username)

# Initialize Dash app
app = Dash(
    __name__, 
    server=server, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    assets_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets'),
    suppress_callback_exceptions=True
)

# Import layout modules
from layouts import download_tab, selection_tab, technical_tab, test_tab, predict_tab, research_tab, aiplus_tab

# Define login layout with improved styling
def get_login_layout():
    return html.Div([
        html.Div([
            html.H2('Sundai Stocks - Login', className="bank-title"),
            html.Div(className="bank-emblem"),
            html.Div([
                html.Label('Username', className="bank-label"),
                dcc.Input(
                    id='username',
                    type='text',
                    className="bank-input",
                    n_submit=0  # Enable Enter key submission
                ),
                html.Label('Password', className="bank-label"),
                dcc.Input(
                    id='password',
                    type='password',
                    className="bank-input",
                    n_submit=0  # Enable Enter key submission
                ),
                html.Button('Login', id='login-button', className="bank-button"),
                html.Div(id='login-output', className="bank-error")
            ], className="bank-login-form")
        ], className="bank-login-container")
    ], className="bank-app")

# Define protected layout
def get_protected_layout():
    return html.Div([
        # Bank Header with Emblem
        html.Header([
            html.Div(className="bank-emblem"),
            html.H1("Sundai Stocks", className="bank-title"),
            html.P("Financial Intelligence Using Math + AI", className="bank-subtitle"),
            html.A('Logout', href='/logout', className='bank-logout-button')
        ], className="bank-header"),
        
        # Main Container
        html.Div([
            # Navigation Tabs
            dcc.Tabs([
                dcc.Tab(label='Download', value='download', className="bank-tab", selected_className="bank-tab--selected"),
                dcc.Tab(label='Screener', value='selection', className="bank-tab", selected_className="bank-tab--selected"),
                dcc.Tab(label='Fetch Data', value='technical', className="bank-tab", selected_className="bank-tab--selected"),
                dcc.Tab(label='Test', value='test', className="bank-tab", selected_className="bank-tab--selected"),
                dcc.Tab(label='Predict', value='predict', className="bank-tab", selected_className="bank-tab--selected"),
                dcc.Tab(label='Research', value='research', className="bank-tab", selected_className="bank-tab--selected"),
                dcc.Tab(
                    label='AI+ ', 
                    value='aiplus', 
                    className="bank-tab", 
                    selected_className="bank-tab--selected"
                )
            ], value='download', id='tabs', className="bank-tabs"),
            
            # Tab content containers with bank theme styling
            html.Div([
                download_tab.layout,
                selection_tab.get_layout(),
                technical_tab.layout,
                test_tab.get_layout(),
                predict_tab.get_layout(),
                research_tab.get_layout(),
                aiplus_tab.get_layout(),
            ], className="bank-content"),
            
            # Bank Footer
            html.Footer([
                html.P("© 2025 Sundai Stocks"),
                html.P("All market data is for informational purposes only")
            ], className="bank-footer")
        ], className="bank-container"),
    ], className="bank-app texture-bg")

# Main layout with URL routing
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# URL callback for page routing - simplified to use Flask-Login directly
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    # Handle logout
    if pathname == '/logout':
        logout_user()
        return get_login_layout()
    
    # Check if user is authenticated
    if current_user.is_authenticated:
        return get_protected_layout()
    else:
        return get_login_layout()

# Login callback that handles both button clicks and Enter key submissions
@app.callback(
    Output('url', 'pathname', allow_duplicate=True),
    Output('login-output', 'children'),
    [Input('login-button', 'n_clicks'),
     Input('username', 'n_submit'),
     Input('password', 'n_submit')],
    [State('username', 'value'),
     State('password', 'value')],
    prevent_initial_call=True
)
def login_callback(n_clicks, username_submit, password_submit, username, password):
    # Check if any trigger happened
    if n_clicks or username_submit or password_submit:
        if not username or not password:
            return no_update, 'Please enter both username and password'
            
        if username == VALID_USERNAME and check_password_hash(VALID_PASSWORD_HASH, password):
            login_user(User(username))
            return '/', ''
        else:
            return no_update, 'Invalid username or password'
    
    return no_update, ''

# Tab switching callback
@app.callback(
    [Output('download-content', 'style'),
     Output('selection-content', 'style'),
     Output('technical-content', 'style'),
     Output('test-content', 'style'),
     Output('predict-content', 'style'),
     Output('research-content', 'style'),
     Output('aiplus-content', 'style')],
    Input('tabs', 'value')
)
def toggle_tab_content(tab):
    """
    Toggle visibility of tab content based on selected tab.
    
    Args:
        tab (str): Selected tab value.
        
    Returns:
        tuple: Display style for each tab content div.
    """
    styles = {'display': 'none'}
    current_style = {'display': 'block', 'animation': 'slide-in 0.4s ease-out forwards'}
    
    return (
        current_style if tab == 'download' else styles,
        current_style if tab == 'selection' else styles,
        current_style if tab == 'technical' else styles,
        current_style if tab == 'test' else styles,
        current_style if tab == 'predict' else styles,
        current_style if tab == 'research' else styles,
        current_style if tab == 'aiplus' else styles,
    )

# Add custom CSS for login screen
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Stock Analysis Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            .bank-login-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 100vh;
                padding: 20px;
                background-color: var(--bank-cream);
                background-image: url('data:image/png;base64,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');
            }
            .bank-login-form {
                display: flex;
                flex-direction: column;
                gap: 15px;
                width: 100%;
                max-width: 400px;
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
                border: 1px solid var(--bank-border);
            }
            .bank-logout-button {
                display: inline-block;
                background-color: var(--bank-error);
                color: white;
                padding: 5px 15px;
                border-radius: 4px;
                text-decoration: none;
                position: absolute;
                top: 20px;
                right: 20px;
                font-family: 'Roboto Slab', serif;
                font-size: 14px;
                font-weight: 600;
                transition: background-color 0.3s;
            }
            .bank-logout-button:hover {
                background-color: #7a2b2b;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Run the app
if __name__ == '__main__':
    # For local development
    app.run_server(debug=True, host='0.0.0.0')