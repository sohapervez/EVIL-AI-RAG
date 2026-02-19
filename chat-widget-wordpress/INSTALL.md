# EVIL-AI RAG -- WordPress Installation Guide

This guide covers two integration options:

1. **Chat Widget only** -- a single `<script>` tag, no plugin needed.
2. **Full Plugin** -- admin UI for managing papers, analytics, and automatic widget injection.

---

## Prerequisites

| Requirement | Minimum |
|---|---|
| WordPress | 5.0+ |
| PHP | 7.4+ |
| EVIL-AI RAG API | Deployed and accessible (e.g. on Rahti) |

You will also need the **API Key** (shared secret) configured in your Rahti deployment. If you have not deployed the API yet, see the main project README.

---

## 1. Chat Widget Setup (standalone, no plugin)

The chat widget is a self-contained JavaScript file served by the API. You only need one `<script>` tag.

### Option A -- Default API URL

Add the following to your theme's `footer.php` (or use a "Header/Footer Scripts" plugin):

```html
<script
  src="https://evil-ai-rag.rahtiapp.fi/static/evil-ai-chat.js"
  defer
></script>
```

The widget auto-detects its API URL from the script `src`.

### Option B -- Custom API URL via data attribute

If your API lives at a different domain:

```html
<script
  src="https://evil-ai-rag.rahtiapp.fi/static/evil-ai-chat.js"
  data-api-url="https://your-api.example.com"
  defer
></script>
```

### Option C -- Custom API URL via JS variable

```html
<script>
  window.EVIL_AI_API_URL = "https://your-api.example.com";
</script>
<script
  src="https://evil-ai-rag.rahtiapp.fi/static/evil-ai-chat.js"
  defer
></script>
```

The widget renders a floating chat button in the bottom-right corner of every page.

---

## 2. Plugin Installation

The plugin gives editors and admins a WordPress dashboard for managing research papers, viewing analytics, and automatically injecting the chat widget into the frontend.

### Step 1 -- Copy files

Copy the entire `evil-ai-doc-manager/` folder into your WordPress plugins directory:

```
wp-content/plugins/evil-ai-doc-manager/
├── evil-ai-doc-manager.php      # Main plugin file
├── includes/
│   ├── class-api-client.php     # HTTP client for the Rahti API
│   └── class-admin-page.php     # Admin page renderer
└── assets/
    ├── admin.css                # Admin styles
    └── admin.js                 # Admin JavaScript
```

You can copy via SFTP, `scp`, or your hosting file manager.

### Step 2 -- Activate

1. Log in to **WordPress Admin**.
2. Go to **Plugins > Installed Plugins**.
3. Find **EVIL-AI Document Manager** in the list.
4. Click **Activate**.

Once activated the plugin automatically injects the chat widget `<script>` tag into every frontend page (using the API URL from Settings).

---

## 3. Plugin Configuration

### Step 1 -- Open Settings

Navigate to **EVIL-AI Docs > Settings** in the WordPress admin sidebar.

### Step 2 -- Enter API credentials

| Field | Example |
|---|---|
| **API URL** | `https://evil-ai-rag.rahtiapp.fi` |
| **API Key** | `your-shared-secret-key` |

- The API URL should be the base URL with **no trailing slash**.
- The API Key is the `ADMIN_API_KEY` value configured in your Rahti deployment environment variables.

### Step 3 -- Test the connection

1. Click **Save Changes**.
2. Click **Test Connection**.
3. You should see a green "Connection successful!" message with the API health response.

---

## 4. Managing Papers

Navigate to **EVIL-AI Docs** in the admin sidebar.

### Viewing papers

The Documents tab shows a table of all indexed papers with their title, filename, and authors. The list is loaded directly from the API.

### Uploading a paper

1. Click the file input and select a PDF (max 50 MB).
2. Click **Upload Paper**.
3. Wait for the upload and indexing to complete.
4. The paper appears in the table automatically.

### Deleting a paper

1. Click the trash icon next to the paper.
2. Confirm the deletion.
3. The paper and all its indexed chunks are removed.

### Reindexing (admin only)

The **Reindex All Papers** button (visible to administrators only) triggers a full reindex of every paper on the server. This is useful after configuration changes or if the vector store becomes inconsistent.

> Reindexing can take several minutes depending on the number and size of papers.

---

## 5. Analytics

Click the **Analytics** tab on the EVIL-AI Docs page.

1. Select a time period from the dropdown (7 days, 30 days, 90 days, or 1 year).
2. Click **Load Analytics**.

The dashboard shows:

- **Summary cards** -- total questions, unique sessions, average response time, error rate.
- **Top Cited Papers** -- horizontal bar chart of the most frequently referenced papers.
- **Questions Per Day** -- daily question volume chart.
- **Recent Questions** -- table with timestamp, question text, papers cited, and response latency.

> Analytics data comes from the API. If the API does not have an analytics endpoint enabled, this tab will show an error.

---

## 6. Troubleshooting

### "Connection failed"

- Verify the **API URL** in Settings is correct and accessible from your WordPress server.
- Ensure the Rahti pod is running: check the Rahti console or run `oc get pods`.
- Make sure there is no firewall blocking outbound HTTPS from your WordPress host.

### "Unauthorized" / 401 error

- Double-check the **API Key** matches the `ADMIN_API_KEY` environment variable in your Rahti deployment.
- The key is case-sensitive.

### "Upload failed"

- Confirm the file is a valid PDF.
- Check the file is under 50 MB.
- Verify `file_uploads = On` and `upload_max_filesize >= 50M` in your PHP configuration.
- Also check `post_max_size >= 50M` in `php.ini`.

### Chat widget not showing on the frontend

- Go to **EVIL-AI Docs > Settings** and confirm the API URL is saved.
- View the page source (Ctrl+U) and search for `evil-ai-chat.js` -- the script tag should be present near `</body>`.
- Open the browser console (F12) and look for JavaScript errors or failed network requests.
- If you see a 404 for the script, verify the API is running and serves the static file.

### CORS errors in the browser console

- Your WordPress domain must be in the API's allowed CORS origins.
- In the Rahti deployment, update the `CORS_ORIGINS` environment variable to include your WordPress domain, e.g.:
  ```
  CORS_ORIGINS=https://your-wordpress-site.com,https://evil-ai.eu
  ```
- Restart the API pod after changing environment variables.

### Plugin pages load slowly

- The plugin makes HTTP requests to the Rahti API. If the API pod is scaled to zero (sleeping), the first request may take 30--60 seconds while the pod starts up.
- Subsequent requests should be fast.

### PHP timeout during upload or reindex

- Increase `max_execution_time` in `php.ini` (recommended: 300 seconds for reindex).
- The plugin already sets a 120-second timeout for uploads and 300 seconds for reindex, but the PHP process itself must also be allowed to run that long.

---

## 7. Permissions

| Action | Required WordPress capability |
|---|---|
| View papers | `edit_posts` (Editors and above) |
| Upload papers | `edit_posts` (Editors and above) |
| Delete papers | `edit_posts` (Editors and above) |
| View analytics | `edit_posts` (Editors and above) |
| Reindex | `manage_options` (Administrators only) |
| Change settings | `manage_options` (Administrators only) |
| Test connection | `manage_options` (Administrators only) |

---

## 8. Uninstallation

1. Deactivate the plugin in **Plugins > Installed Plugins**.
2. Delete the plugin (this removes the files).
3. The `evil_ai_api_url` and `evil_ai_api_key` options remain in the `wp_options` table. To remove them, run:
   ```sql
   DELETE FROM wp_options WHERE option_name IN ('evil_ai_api_url', 'evil_ai_api_key');
   ```
   Or add an `uninstall.php` file to handle this automatically.
