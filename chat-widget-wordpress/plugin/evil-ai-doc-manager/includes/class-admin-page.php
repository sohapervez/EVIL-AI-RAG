<?php
/**
 * EVIL-AI Admin Page Renderer.
 *
 * Outputs the HTML for the Documents and Settings admin pages.
 *
 * @package Evil_AI_Doc_Manager
 */

// Prevent direct access.
if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

/**
 * Class Evil_AI_Admin_Page
 */
class Evil_AI_Admin_Page {

    /**
     * Singleton instance.
     *
     * @var Evil_AI_Admin_Page|null
     */
    private static $instance = null;

    /**
     * Get singleton instance.
     *
     * @return Evil_AI_Admin_Page
     */
    public static function get_instance() {
        if ( null === self::$instance ) {
            self::$instance = new self();
        }
        return self::$instance;
    }

    /**
     * Private constructor.
     */
    private function __construct() {}

    // -------------------------------------------------------------------------
    // Documents Page
    // -------------------------------------------------------------------------

    /**
     * Render the main Documents management page.
     */
    public function render_documents_page() {
        $api_url = get_option( 'evil_ai_api_url', '' );
        ?>
        <div class="wrap evil-ai-wrap">
            <h1><?php esc_html_e( 'EVIL-AI Document Manager', 'evil-ai-doc-manager' ); ?></h1>

            <?php if ( empty( $api_url ) ) : ?>
                <div class="notice notice-warning">
                    <p>
                        <?php
                        printf(
                            /* translators: %s: URL to the settings page */
                            esc_html__( 'API URL is not configured. Please visit the %s page first.', 'evil-ai-doc-manager' ),
                            '<a href="' . esc_url( admin_url( 'admin.php?page=evil-ai-settings' ) ) . '">' . esc_html__( 'Settings', 'evil-ai-doc-manager' ) . '</a>'
                        );
                        ?>
                    </p>
                </div>
            <?php endif; ?>

            <!-- Notification area -->
            <div id="evil-ai-notifications"></div>

            <!-- Tab navigation -->
            <nav class="evil-ai-tabs">
                <a href="#documents" class="evil-ai-tab-link active" data-tab="documents">
                    <span class="dashicons dashicons-media-document"></span>
                    <?php esc_html_e( 'Documents', 'evil-ai-doc-manager' ); ?>
                </a>
                <a href="#analytics" class="evil-ai-tab-link" data-tab="analytics">
                    <span class="dashicons dashicons-chart-bar"></span>
                    <?php esc_html_e( 'Analytics', 'evil-ai-doc-manager' ); ?>
                </a>
            </nav>

            <!-- Documents Tab -->
            <div id="evil-ai-tab-documents" class="evil-ai-tab-content active">

                <!-- Upload Form -->
                <div class="evil-ai-upload-section">
                    <h2><?php esc_html_e( 'Upload Paper', 'evil-ai-doc-manager' ); ?></h2>
                    <form id="evil-ai-upload-form" enctype="multipart/form-data">
                        <div class="evil-ai-upload-row">
                            <input
                                type="file"
                                id="evil-ai-file-input"
                                name="paper"
                                accept=".pdf"
                                required
                            />
                            <button type="submit" class="button button-primary" id="evil-ai-upload-btn">
                                <span class="dashicons dashicons-upload"></span>
                                <?php esc_html_e( 'Upload Paper', 'evil-ai-doc-manager' ); ?>
                            </button>
                            <span class="spinner" id="evil-ai-upload-spinner"></span>
                        </div>
                        <p class="description">
                            <?php esc_html_e( 'Select a PDF file (max 50 MB). The paper will be uploaded and indexed automatically.', 'evil-ai-doc-manager' ); ?>
                        </p>
                    </form>
                </div>

                <!-- Actions Bar -->
                <div class="evil-ai-actions-bar">
                    <button type="button" class="button" id="evil-ai-refresh-btn">
                        <span class="dashicons dashicons-update"></span>
                        <?php esc_html_e( 'Refresh', 'evil-ai-doc-manager' ); ?>
                    </button>

                    <?php if ( current_user_can( 'manage_options' ) ) : ?>
                        <button type="button" class="button button-secondary" id="evil-ai-reindex-btn">
                            <span class="dashicons dashicons-image-rotate"></span>
                            <?php esc_html_e( 'Reindex All Papers', 'evil-ai-doc-manager' ); ?>
                        </button>
                    <?php endif; ?>

                    <span class="spinner" id="evil-ai-action-spinner"></span>
                </div>

                <!-- Papers Table -->
                <div id="evil-ai-papers-container">
                    <table class="wp-list-table widefat fixed striped" id="evil-ai-papers-table">
                        <thead>
                            <tr>
                                <th class="column-number" scope="col">#</th>
                                <th class="column-title" scope="col"><?php esc_html_e( 'Title', 'evil-ai-doc-manager' ); ?></th>
                                <th class="column-filename" scope="col"><?php esc_html_e( 'Filename', 'evil-ai-doc-manager' ); ?></th>
                                <th class="column-authors" scope="col"><?php esc_html_e( 'Authors', 'evil-ai-doc-manager' ); ?></th>
                                <th class="column-actions" scope="col"><?php esc_html_e( 'Actions', 'evil-ai-doc-manager' ); ?></th>
                            </tr>
                        </thead>
                        <tbody id="evil-ai-papers-tbody">
                            <tr class="evil-ai-loading-row">
                                <td colspan="5">
                                    <span class="spinner is-active" style="float:none;"></span>
                                    <?php esc_html_e( 'Loading papers...', 'evil-ai-doc-manager' ); ?>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Analytics Tab -->
            <div id="evil-ai-tab-analytics" class="evil-ai-tab-content">

                <!-- Period Selector -->
                <div class="evil-ai-analytics-controls">
                    <label for="evil-ai-analytics-days"><?php esc_html_e( 'Period:', 'evil-ai-doc-manager' ); ?></label>
                    <select id="evil-ai-analytics-days">
                        <option value="7"><?php esc_html_e( 'Last 7 days', 'evil-ai-doc-manager' ); ?></option>
                        <option value="30" selected><?php esc_html_e( 'Last 30 days', 'evil-ai-doc-manager' ); ?></option>
                        <option value="90"><?php esc_html_e( 'Last 90 days', 'evil-ai-doc-manager' ); ?></option>
                        <option value="365"><?php esc_html_e( 'Last year', 'evil-ai-doc-manager' ); ?></option>
                    </select>
                    <button type="button" class="button" id="evil-ai-load-analytics-btn">
                        <span class="dashicons dashicons-chart-bar"></span>
                        <?php esc_html_e( 'Load Analytics', 'evil-ai-doc-manager' ); ?>
                    </button>
                    <span class="spinner" id="evil-ai-analytics-spinner"></span>
                </div>

                <!-- Summary Cards -->
                <div class="evil-ai-summary-cards" id="evil-ai-summary-cards">
                    <div class="evil-ai-card">
                        <div class="evil-ai-card-icon dashicons dashicons-format-chat"></div>
                        <div class="evil-ai-card-body">
                            <span class="evil-ai-card-value" id="evil-ai-total-questions">--</span>
                            <span class="evil-ai-card-label"><?php esc_html_e( 'Total Questions', 'evil-ai-doc-manager' ); ?></span>
                        </div>
                    </div>
                    <div class="evil-ai-card">
                        <div class="evil-ai-card-icon dashicons dashicons-groups"></div>
                        <div class="evil-ai-card-body">
                            <span class="evil-ai-card-value" id="evil-ai-unique-sessions">--</span>
                            <span class="evil-ai-card-label"><?php esc_html_e( 'Unique Sessions', 'evil-ai-doc-manager' ); ?></span>
                        </div>
                    </div>
                    <div class="evil-ai-card">
                        <div class="evil-ai-card-icon dashicons dashicons-clock"></div>
                        <div class="evil-ai-card-body">
                            <span class="evil-ai-card-value" id="evil-ai-avg-response-time">--</span>
                            <span class="evil-ai-card-label"><?php esc_html_e( 'Avg Response Time', 'evil-ai-doc-manager' ); ?></span>
                        </div>
                    </div>
                    <div class="evil-ai-card">
                        <div class="evil-ai-card-icon dashicons dashicons-warning"></div>
                        <div class="evil-ai-card-body">
                            <span class="evil-ai-card-value" id="evil-ai-error-rate">--</span>
                            <span class="evil-ai-card-label"><?php esc_html_e( 'Error Rate', 'evil-ai-doc-manager' ); ?></span>
                        </div>
                    </div>
                </div>

                <!-- Top Cited Papers -->
                <div class="evil-ai-analytics-section">
                    <h3><?php esc_html_e( 'Top Cited Papers', 'evil-ai-doc-manager' ); ?></h3>
                    <div id="evil-ai-top-papers" class="evil-ai-bar-chart">
                        <p class="evil-ai-placeholder"><?php esc_html_e( 'Click "Load Analytics" to view data.', 'evil-ai-doc-manager' ); ?></p>
                    </div>
                </div>

                <!-- Questions Per Day -->
                <div class="evil-ai-analytics-section">
                    <h3><?php esc_html_e( 'Questions Per Day', 'evil-ai-doc-manager' ); ?></h3>
                    <div id="evil-ai-questions-per-day" class="evil-ai-bar-chart">
                        <p class="evil-ai-placeholder"><?php esc_html_e( 'Click "Load Analytics" to view data.', 'evil-ai-doc-manager' ); ?></p>
                    </div>
                </div>

                <!-- Recent Questions -->
                <div class="evil-ai-analytics-section">
                    <h3><?php esc_html_e( 'Recent Questions', 'evil-ai-doc-manager' ); ?></h3>
                    <table class="wp-list-table widefat fixed striped" id="evil-ai-recent-questions-table">
                        <thead>
                            <tr>
                                <th class="column-timestamp" scope="col"><?php esc_html_e( 'Timestamp', 'evil-ai-doc-manager' ); ?></th>
                                <th class="column-question" scope="col"><?php esc_html_e( 'Question', 'evil-ai-doc-manager' ); ?></th>
                                <th class="column-papers-cited" scope="col"><?php esc_html_e( 'Papers Cited', 'evil-ai-doc-manager' ); ?></th>
                                <th class="column-latency" scope="col"><?php esc_html_e( 'Latency', 'evil-ai-doc-manager' ); ?></th>
                            </tr>
                        </thead>
                        <tbody id="evil-ai-recent-questions-tbody">
                            <tr>
                                <td colspan="4" class="evil-ai-placeholder">
                                    <?php esc_html_e( 'Click "Load Analytics" to view data.', 'evil-ai-doc-manager' ); ?>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        <?php
    }

    // -------------------------------------------------------------------------
    // Settings Page
    // -------------------------------------------------------------------------

    /**
     * Render the Settings page.
     */
    public function render_settings_page() {
        ?>
        <div class="wrap evil-ai-wrap">
            <h1><?php esc_html_e( 'EVIL-AI Settings', 'evil-ai-doc-manager' ); ?></h1>

            <div id="evil-ai-notifications"></div>

            <form method="post" action="options.php">
                <?php
                settings_fields( 'evil_ai_settings_group' );
                do_settings_sections( 'evil-ai-settings' );
                submit_button();
                ?>
            </form>

            <hr />

            <h2><?php esc_html_e( 'Connection Test', 'evil-ai-doc-manager' ); ?></h2>
            <p class="description">
                <?php esc_html_e( 'Save your settings first, then click the button below to verify the connection to the API.', 'evil-ai-doc-manager' ); ?>
            </p>
            <p>
                <button type="button" class="button button-secondary" id="evil-ai-test-connection">
                    <span class="dashicons dashicons-networking"></span>
                    <?php esc_html_e( 'Test Connection', 'evil-ai-doc-manager' ); ?>
                </button>
                <span class="spinner" id="evil-ai-test-spinner"></span>
            </p>
            <div id="evil-ai-test-result"></div>
        </div>
        <?php
    }
}
