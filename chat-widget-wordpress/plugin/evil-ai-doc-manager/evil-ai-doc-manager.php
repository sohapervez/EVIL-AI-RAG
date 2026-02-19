<?php
/**
 * Plugin Name: EVIL-AI Document Manager
 * Plugin URI: https://evil-ai.eu
 * Description: Manage research papers for the EVIL-AI RAG chatbot. Upload, delete, and reindex PDFs via the Rahti API.
 * Version: 1.0.0
 * Author: EVIL-AI Team
 * License: GPL v2 or later
 * Text Domain: evil-ai-doc-manager
 */

// Prevent direct access.
if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

define( 'EVIL_AI_DOC_MANAGER_VERSION', '1.0.0' );
define( 'EVIL_AI_DOC_MANAGER_FILE', __FILE__ );
define( 'EVIL_AI_DOC_MANAGER_DIR', plugin_dir_path( __FILE__ ) );
define( 'EVIL_AI_DOC_MANAGER_URL', plugin_dir_url( __FILE__ ) );

// Include required files.
require_once EVIL_AI_DOC_MANAGER_DIR . 'includes/class-api-client.php';
require_once EVIL_AI_DOC_MANAGER_DIR . 'includes/class-admin-page.php';

/**
 * Main plugin class.
 */
class Evil_AI_Doc_Manager {

    /**
     * Singleton instance.
     *
     * @var Evil_AI_Doc_Manager|null
     */
    private static $instance = null;

    /**
     * Get singleton instance.
     *
     * @return Evil_AI_Doc_Manager
     */
    public static function get_instance() {
        if ( null === self::$instance ) {
            self::$instance = new self();
        }
        return self::$instance;
    }

    /**
     * Constructor.
     */
    private function __construct() {
        add_action( 'admin_menu', array( $this, 'register_admin_menu' ) );
        add_action( 'admin_init', array( $this, 'register_settings' ) );
        add_action( 'admin_enqueue_scripts', array( $this, 'enqueue_admin_assets' ) );
        add_action( 'wp_footer', array( $this, 'inject_chat_widget' ) );

        // AJAX handlers.
        add_action( 'wp_ajax_evil_ai_upload_paper', array( $this, 'ajax_upload_paper' ) );
        add_action( 'wp_ajax_evil_ai_delete_paper', array( $this, 'ajax_delete_paper' ) );
        add_action( 'wp_ajax_evil_ai_reindex', array( $this, 'ajax_reindex' ) );
        add_action( 'wp_ajax_evil_ai_test_connection', array( $this, 'ajax_test_connection' ) );
        add_action( 'wp_ajax_evil_ai_get_analytics', array( $this, 'ajax_get_analytics' ) );
    }

    /**
     * Register admin menu pages.
     */
    public function register_admin_menu() {
        add_menu_page(
            __( 'EVIL-AI Documents', 'evil-ai-doc-manager' ),
            __( 'EVIL-AI Docs', 'evil-ai-doc-manager' ),
            'edit_posts',
            'evil-ai-docs',
            array( Evil_AI_Admin_Page::get_instance(), 'render_documents_page' ),
            'dashicons-media-document',
            30
        );

        add_submenu_page(
            'evil-ai-docs',
            __( 'EVIL-AI Settings', 'evil-ai-doc-manager' ),
            __( 'Settings', 'evil-ai-doc-manager' ),
            'manage_options',
            'evil-ai-settings',
            array( Evil_AI_Admin_Page::get_instance(), 'render_settings_page' )
        );
    }

    /**
     * Register plugin settings.
     */
    public function register_settings() {
        register_setting(
            'evil_ai_settings_group',
            'evil_ai_api_url',
            array(
                'type'              => 'string',
                'sanitize_callback' => 'esc_url_raw',
                'default'           => '',
            )
        );

        register_setting(
            'evil_ai_settings_group',
            'evil_ai_api_key',
            array(
                'type'              => 'string',
                'sanitize_callback' => 'sanitize_text_field',
                'default'           => '',
            )
        );

        add_settings_section(
            'evil_ai_main_section',
            __( 'API Configuration', 'evil-ai-doc-manager' ),
            array( $this, 'render_settings_section' ),
            'evil-ai-settings'
        );

        add_settings_field(
            'evil_ai_api_url',
            __( 'API URL', 'evil-ai-doc-manager' ),
            array( $this, 'render_api_url_field' ),
            'evil-ai-settings',
            'evil_ai_main_section'
        );

        add_settings_field(
            'evil_ai_api_key',
            __( 'API Key', 'evil-ai-doc-manager' ),
            array( $this, 'render_api_key_field' ),
            'evil-ai-settings',
            'evil_ai_main_section'
        );
    }

    /**
     * Render the settings section description.
     */
    public function render_settings_section() {
        echo '<p>' . esc_html__( 'Configure the connection to your EVIL-AI RAG API backend.', 'evil-ai-doc-manager' ) . '</p>';
    }

    /**
     * Render the API URL field.
     */
    public function render_api_url_field() {
        $value = get_option( 'evil_ai_api_url', '' );
        printf(
            '<input type="url" id="evil_ai_api_url" name="evil_ai_api_url" value="%s" class="regular-text" placeholder="https://evil-ai-rag.rahtiapp.fi" />',
            esc_attr( $value )
        );
        echo '<p class="description">' . esc_html__( 'The full URL of your Rahti-hosted EVIL-AI RAG API (no trailing slash).', 'evil-ai-doc-manager' ) . '</p>';
    }

    /**
     * Render the API Key field.
     */
    public function render_api_key_field() {
        $value = get_option( 'evil_ai_api_key', '' );
        printf(
            '<input type="password" id="evil_ai_api_key" name="evil_ai_api_key" value="%s" class="regular-text" autocomplete="new-password" />',
            esc_attr( $value )
        );
        echo '<p class="description">' . esc_html__( 'The shared-secret API key configured in the Rahti deployment.', 'evil-ai-doc-manager' ) . '</p>';
    }

    /**
     * Enqueue admin CSS and JS only on plugin pages.
     *
     * @param string $hook_suffix The current admin page hook suffix.
     */
    public function enqueue_admin_assets( $hook_suffix ) {
        $plugin_pages = array(
            'toplevel_page_evil-ai-docs',
            'evil-ai-docs_page_evil-ai-settings',
        );

        if ( ! in_array( $hook_suffix, $plugin_pages, true ) ) {
            return;
        }

        wp_enqueue_style(
            'evil-ai-admin-css',
            EVIL_AI_DOC_MANAGER_URL . 'assets/admin.css',
            array(),
            EVIL_AI_DOC_MANAGER_VERSION
        );

        wp_enqueue_script(
            'evil-ai-admin-js',
            EVIL_AI_DOC_MANAGER_URL . 'assets/admin.js',
            array( 'jquery' ),
            EVIL_AI_DOC_MANAGER_VERSION,
            true
        );

        wp_localize_script( 'evil-ai-admin-js', 'evil_ai_admin', array(
            'ajax_url' => admin_url( 'admin-ajax.php' ),
            'nonce'    => wp_create_nonce( 'evil_ai_nonce' ),
            'api_url'  => get_option( 'evil_ai_api_url', '' ),
        ) );
    }

    /**
     * Inject the chat widget script into the frontend footer.
     */
    public function inject_chat_widget() {
        $api_url = get_option( 'evil_ai_api_url', '' );

        if ( empty( $api_url ) ) {
            return;
        }

        $script_url = rtrim( $api_url, '/' ) . '/static/evil-ai-chat.js';

        printf(
            '<script src="%s" data-api-url="%s" defer></script>' . "\n",
            esc_url( $script_url ),
            esc_url( $api_url )
        );
    }

    // -------------------------------------------------------------------------
    // AJAX Handlers
    // -------------------------------------------------------------------------

    /**
     * Handle paper upload via AJAX.
     */
    public function ajax_upload_paper() {
        check_ajax_referer( 'evil_ai_nonce', 'nonce' );

        if ( ! current_user_can( 'edit_posts' ) ) {
            wp_send_json_error( array( 'message' => __( 'You do not have permission to upload papers.', 'evil-ai-doc-manager' ) ), 403 );
        }

        if ( empty( $_FILES['paper'] ) ) {
            wp_send_json_error( array( 'message' => __( 'No file was uploaded.', 'evil-ai-doc-manager' ) ) );
        }

        $file = $_FILES['paper'];

        // Validate file type.
        $file_type = wp_check_filetype( $file['name'], array( 'pdf' => 'application/pdf' ) );
        if ( empty( $file_type['ext'] ) ) {
            wp_send_json_error( array( 'message' => __( 'Only PDF files are allowed.', 'evil-ai-doc-manager' ) ) );
        }

        // Validate file size (50 MB max).
        $max_size = 50 * 1024 * 1024;
        if ( $file['size'] > $max_size ) {
            wp_send_json_error( array( 'message' => __( 'File exceeds the 50 MB size limit.', 'evil-ai-doc-manager' ) ) );
        }

        // Validate no upload errors.
        if ( UPLOAD_ERR_OK !== $file['error'] ) {
            wp_send_json_error( array( 'message' => __( 'File upload error. Please try again.', 'evil-ai-doc-manager' ) ) );
        }

        $client   = new Evil_AI_API_Client();
        $response = $client->upload_paper( $file['tmp_name'], $file['name'] );

        if ( isset( $response['error'] ) ) {
            wp_send_json_error( array( 'message' => $response['error'] ) );
        }

        wp_send_json_success( $response );
    }

    /**
     * Handle paper deletion via AJAX.
     */
    public function ajax_delete_paper() {
        check_ajax_referer( 'evil_ai_nonce', 'nonce' );

        if ( ! current_user_can( 'edit_posts' ) ) {
            wp_send_json_error( array( 'message' => __( 'You do not have permission to delete papers.', 'evil-ai-doc-manager' ) ), 403 );
        }

        $filename = isset( $_POST['filename'] ) ? sanitize_file_name( $_POST['filename'] ) : '';

        if ( empty( $filename ) ) {
            wp_send_json_error( array( 'message' => __( 'No filename provided.', 'evil-ai-doc-manager' ) ) );
        }

        $client   = new Evil_AI_API_Client();
        $response = $client->delete_paper( $filename );

        if ( isset( $response['error'] ) ) {
            wp_send_json_error( array( 'message' => $response['error'] ) );
        }

        wp_send_json_success( $response );
    }

    /**
     * Handle reindex via AJAX (admin only).
     */
    public function ajax_reindex() {
        check_ajax_referer( 'evil_ai_nonce', 'nonce' );

        if ( ! current_user_can( 'manage_options' ) ) {
            wp_send_json_error( array( 'message' => __( 'Only administrators can trigger a reindex.', 'evil-ai-doc-manager' ) ), 403 );
        }

        $client   = new Evil_AI_API_Client();
        $response = $client->reindex();

        if ( isset( $response['error'] ) ) {
            wp_send_json_error( array( 'message' => $response['error'] ) );
        }

        wp_send_json_success( $response );
    }

    /**
     * Handle connection test via AJAX (admin only).
     */
    public function ajax_test_connection() {
        check_ajax_referer( 'evil_ai_nonce', 'nonce' );

        if ( ! current_user_can( 'manage_options' ) ) {
            wp_send_json_error( array( 'message' => __( 'Only administrators can test the connection.', 'evil-ai-doc-manager' ) ), 403 );
        }

        $client   = new Evil_AI_API_Client();
        $response = $client->health();

        if ( isset( $response['error'] ) ) {
            wp_send_json_error( array( 'message' => $response['error'] ) );
        }

        wp_send_json_success( $response );
    }

    /**
     * Handle analytics retrieval via AJAX.
     */
    public function ajax_get_analytics() {
        check_ajax_referer( 'evil_ai_nonce', 'nonce' );

        if ( ! current_user_can( 'edit_posts' ) ) {
            wp_send_json_error( array( 'message' => __( 'You do not have permission to view analytics.', 'evil-ai-doc-manager' ) ), 403 );
        }

        $days     = isset( $_POST['days'] ) ? absint( $_POST['days'] ) : 30;
        $days     = max( 1, min( 365, $days ) );
        $client   = new Evil_AI_API_Client();
        $response = $client->get_analytics( $days );

        if ( isset( $response['error'] ) ) {
            wp_send_json_error( array( 'message' => $response['error'] ) );
        }

        wp_send_json_success( $response );
    }
}

// Initialize the plugin.
Evil_AI_Doc_Manager::get_instance();
