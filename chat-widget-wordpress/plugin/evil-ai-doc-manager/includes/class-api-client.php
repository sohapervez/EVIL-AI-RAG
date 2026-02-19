<?php
/**
 * EVIL-AI API Client.
 *
 * Wraps all HTTP communication with the Rahti-hosted EVIL-AI RAG API
 * using WordPress's built-in wp_remote_request() functions.
 *
 * @package Evil_AI_Doc_Manager
 */

// Prevent direct access.
if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

/**
 * Class Evil_AI_API_Client
 */
class Evil_AI_API_Client {

    /**
     * Base API URL (no trailing slash).
     *
     * @var string
     */
    private $api_url;

    /**
     * API key (Bearer token).
     *
     * @var string
     */
    private $api_key;

    /**
     * Constructor. Reads settings from wp_options.
     */
    public function __construct() {
        $this->api_url = rtrim( get_option( 'evil_ai_api_url', '' ), '/' );
        $this->api_key = get_option( 'evil_ai_api_key', '' );
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /**
     * Execute an HTTP request against the API.
     *
     * @param string $method   HTTP method (GET, POST, DELETE, etc.).
     * @param string $endpoint API endpoint path (e.g. /api/v1/health).
     * @param array  $args     Optional extra arguments for wp_remote_request().
     *
     * @return array Decoded JSON response or array with 'error' key on failure.
     */
    private function request( $method, $endpoint, $args = array() ) {
        if ( empty( $this->api_url ) ) {
            return array( 'error' => __( 'API URL is not configured. Please update the settings.', 'evil-ai-doc-manager' ) );
        }

        $url = $this->api_url . $endpoint;

        $defaults = array(
            'method'  => $method,
            'timeout' => 60,
            'headers' => array(
                'Authorization' => 'Bearer ' . $this->api_key,
                'Accept'        => 'application/json',
            ),
        );

        $args = wp_parse_args( $args, $defaults );

        // Merge additional headers with defaults.
        if ( isset( $args['headers'] ) && isset( $defaults['headers'] ) ) {
            $args['headers'] = wp_parse_args( $args['headers'], $defaults['headers'] );
        }

        $response = wp_remote_request( $url, $args );

        if ( is_wp_error( $response ) ) {
            return array( 'error' => $response->get_error_message() );
        }

        $code = wp_remote_retrieve_response_code( $response );
        $body = wp_remote_retrieve_body( $response );
        $data = json_decode( $body, true );

        if ( $code >= 400 ) {
            $message = isset( $data['detail'] ) ? $data['detail'] : sprintf(
                /* translators: %d: HTTP status code */
                __( 'HTTP error %d', 'evil-ai-doc-manager' ),
                $code
            );
            return array( 'error' => $message );
        }

        if ( null === $data ) {
            // The response was not valid JSON -- return the raw body.
            return array( 'raw' => $body );
        }

        return $data;
    }

    // -------------------------------------------------------------------------
    // Public API methods
    // -------------------------------------------------------------------------

    /**
     * Health check.
     *
     * @return array API health status.
     */
    public function health() {
        return $this->request( 'GET', '/api/v1/health' );
    }

    /**
     * Retrieve the list of indexed papers.
     *
     * @return array List of paper objects.
     */
    public function get_papers() {
        return $this->request( 'GET', '/api/v1/papers' );
    }

    /**
     * Upload a PDF paper to the API.
     *
     * Uses a multipart/form-data POST built with WordPress's HTTP API.
     *
     * @param string $file_path Absolute path to the temporary uploaded file.
     * @param string $filename  Original filename (e.g. paper.pdf).
     *
     * @return array API response or array with 'error' key.
     */
    public function upload_paper( $file_path, $filename ) {
        if ( empty( $this->api_url ) ) {
            return array( 'error' => __( 'API URL is not configured. Please update the settings.', 'evil-ai-doc-manager' ) );
        }

        $url = $this->api_url . '/api/v1/papers';

        // Build a multipart boundary.
        $boundary = wp_generate_password( 24, false );

        // Read the file contents.
        $file_contents = file_get_contents( $file_path ); // phpcs:ignore WordPress.WP.AlternativeFunctions.file_get_contents_file_get_contents
        if ( false === $file_contents ) {
            return array( 'error' => __( 'Could not read the uploaded file.', 'evil-ai-doc-manager' ) );
        }

        // Build multipart body.
        $filename = str_replace( array( '"', "\r", "\n", "\0" ), '', basename( $filename ) );
        $body  = '';
        $body .= '--' . $boundary . "\r\n";
        $body .= 'Content-Disposition: form-data; name="file"; filename="' . $filename . '"' . "\r\n";
        $body .= 'Content-Type: application/pdf' . "\r\n\r\n";
        $body .= $file_contents . "\r\n";
        $body .= '--' . $boundary . '--' . "\r\n";

        $response = wp_remote_post(
            $url,
            array(
                'method'  => 'POST',
                'timeout' => 120,
                'headers' => array(
                    'Authorization' => 'Bearer ' . $this->api_key,
                    'Content-Type'  => 'multipart/form-data; boundary=' . $boundary,
                    'Accept'        => 'application/json',
                ),
                'body'    => $body,
            )
        );

        if ( is_wp_error( $response ) ) {
            return array( 'error' => $response->get_error_message() );
        }

        $code = wp_remote_retrieve_response_code( $response );
        $resp_body = wp_remote_retrieve_body( $response );
        $data = json_decode( $resp_body, true );

        if ( $code >= 400 ) {
            $message = isset( $data['detail'] ) ? $data['detail'] : sprintf(
                /* translators: %d: HTTP status code */
                __( 'HTTP error %d', 'evil-ai-doc-manager' ),
                $code
            );
            return array( 'error' => $message );
        }

        if ( null === $data ) {
            return array( 'raw' => $resp_body );
        }

        return $data;
    }

    /**
     * Delete a paper from the index.
     *
     * @param string $filename The filename to delete.
     *
     * @return array API response.
     */
    public function delete_paper( $filename ) {
        $endpoint = '/api/v1/papers/' . rawurlencode( $filename );
        return $this->request( 'DELETE', $endpoint );
    }

    /**
     * Trigger a full reindex of all papers.
     *
     * @return array API response.
     */
    public function reindex() {
        return $this->request( 'POST', '/api/v1/reindex', array(
            'timeout' => 300,
        ) );
    }

    /**
     * Retrieve analytics data.
     *
     * @param int $days Number of days to query (default 30).
     *
     * @return array Analytics data.
     */
    public function get_analytics( $days = 30 ) {
        $days = absint( $days );
        if ( $days < 1 ) {
            $days = 30;
        }
        return $this->request( 'GET', '/api/v1/analytics?days=' . $days );
    }
}
