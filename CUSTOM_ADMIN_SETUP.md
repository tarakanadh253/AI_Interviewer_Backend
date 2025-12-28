# Custom Admin Interface Setup

## Overview
A custom Django admin interface has been created to match the frontend design with:
- Dark theme with gradient background
- Neon cyan/pink/purple color scheme
- Modern card-based layout
- Glowing effects and smooth transitions
- Responsive design

## Files Created

### Templates
- `templates/admin/base.html` - Main admin base template
- `templates/admin/base_site.html` - Site branding template
- `templates/admin/index.html` - Custom dashboard index

### Static Files
- `static/admin/css/custom-admin.css` - Custom admin stylesheet

## Settings Updated
- Added `templates` directory to `TEMPLATES['DIRS']`
- Added `static` directory to `STATICFILES_DIRS`

## Features

### Design Elements
- **Background**: Dark gradient (hsl(240, 30%, 8%) to hsl(240, 35%, 6%))
- **Cards**: Semi-transparent with backdrop blur
- **Primary Color**: Cyan (hsl(180, 100%, 50%))
- **Secondary Color**: Pink (hsl(330, 100%, 50%))
- **Accent Color**: Purple (hsl(280, 80%, 60%))
- **Glow Effects**: Neon-style box shadows

### Styled Components
- Header with gradient logo
- Navigation breadcrumbs
- Form inputs with focus states
- Buttons with glow effects
- Tables with hover states
- Messages with colored borders
- Pagination with styled links
- Filters sidebar

## Usage

1. **Collect Static Files** (if needed):
   ```bash
   python manage.py collectstatic
   ```

2. **Access Admin**:
   - Navigate to: `http://localhost:8000/admin/`
   - Login with your superuser credentials
   - The custom design will be applied automatically

## Customization

To customize further, edit:
- `static/admin/css/custom-admin.css` - For styling
- `templates/admin/base.html` - For structure
- `templates/admin/base_site.html` - For branding

## Notes

- The design matches the frontend's dark theme
- All colors use HSL format for consistency
- Responsive design works on mobile devices
- Glow effects add a modern, futuristic feel
