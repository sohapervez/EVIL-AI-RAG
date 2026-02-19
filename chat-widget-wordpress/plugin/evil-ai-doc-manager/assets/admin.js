/**
 * EVIL-AI Document Manager - Admin JavaScript
 *
 * Handles all AJAX interactions for the WordPress admin pages:
 * paper listing, upload, deletion, reindex, analytics, and connection testing.
 *
 * @package Evil_AI_Doc_Manager
 */

/* global jQuery, evil_ai_admin */

jQuery( document ).ready( function( $ ) {
    'use strict';

    // -------------------------------------------------------------------------
    // Tab Switching
    // -------------------------------------------------------------------------

    $( '.evil-ai-tab-link' ).on( 'click', function( e ) {
        e.preventDefault();

        var tab = $( this ).data( 'tab' );

        // Update tab links.
        $( '.evil-ai-tab-link' ).removeClass( 'active' );
        $( this ).addClass( 'active' );

        // Update tab panels.
        $( '.evil-ai-tab-content' ).removeClass( 'active' );
        $( '#evil-ai-tab-' + tab ).addClass( 'active' );

        // Auto-load analytics when switching to that tab.
        if ( tab === 'analytics' && $( '#evil-ai-total-questions' ).text() === '--' ) {
            loadAnalytics();
        }
    } );

    // -------------------------------------------------------------------------
    // Papers: Load
    // -------------------------------------------------------------------------

    function loadPapers() {
        var $tbody  = $( '#evil-ai-papers-tbody' );
        var $spinner = $( '#evil-ai-action-spinner' );

        $tbody.html(
            '<tr class="evil-ai-loading-row"><td colspan="5">' +
            '<span class="spinner is-active" style="float:none;"></span> Loading papers...' +
            '</td></tr>'
        );

        // Direct API call for paper listing.
        if ( ! evil_ai_admin.api_url ) {
            $tbody.html(
                '<tr class="evil-ai-empty-row"><td colspan="5">' +
                'API URL not configured. Please visit Settings.' +
                '</td></tr>'
            );
            return;
        }

        $.ajax( {
            url: evil_ai_admin.api_url + '/api/v1/papers',
            type: 'GET',
            headers: {
                'Authorization': 'Bearer ' + evil_ai_admin.api_key,
                'Accept': 'application/json'
            },
            timeout: 30000,
            success: function( response ) {
                renderPapersTable( response );
            },
            error: function( xhr ) {
                var msg = 'Failed to load papers.';
                if ( xhr.responseJSON && xhr.responseJSON.detail ) {
                    msg = xhr.responseJSON.detail;
                } else if ( xhr.status === 0 ) {
                    msg = 'Cannot reach the API. Check the URL and CORS settings.';
                }
                $tbody.html(
                    '<tr class="evil-ai-empty-row"><td colspan="5">' +
                    escapeHtml( msg ) +
                    '</td></tr>'
                );
            }
        } );
    }

    /**
     * Render the papers into the table body.
     *
     * @param {Array|Object} data API response (array of paper objects or object with papers key).
     */
    function renderPapersTable( data ) {
        var $tbody = $( '#evil-ai-papers-tbody' );
        var papers = Array.isArray( data ) ? data : ( data.papers || data.data || [] );

        if ( ! papers.length ) {
            $tbody.html(
                '<tr class="evil-ai-empty-row"><td colspan="5">' +
                'No papers found. Upload a PDF to get started.' +
                '</td></tr>'
            );
            return;
        }

        var rows = '';
        $.each( papers, function( index, paper ) {
            var title    = paper.title || paper.name || '(untitled)';
            var filename = paper.filename || paper.file || paper.name || '';
            var authors  = paper.authors || '';

            rows += '<tr>';
            rows += '<td>' + ( index + 1 ) + '</td>';
            rows += '<td>' + escapeHtml( title ) + '</td>';
            rows += '<td><code>' + escapeHtml( filename ) + '</code></td>';
            rows += '<td>' + escapeHtml( authors ) + '</td>';
            rows += '<td class="column-actions">';
            rows += '<button type="button" class="button button-small evil-ai-delete-btn" data-filename="' + escapeAttr( filename ) + '" title="Delete paper">';
            rows += '<span class="dashicons dashicons-trash"></span>';
            rows += '</button>';
            rows += '</td>';
            rows += '</tr>';
        } );

        $tbody.html( rows );
    }

    // -------------------------------------------------------------------------
    // Papers: Upload
    // -------------------------------------------------------------------------

    $( '#evil-ai-upload-form' ).on( 'submit', function( e ) {
        e.preventDefault();

        var fileInput = $( '#evil-ai-file-input' )[0];
        if ( ! fileInput.files || ! fileInput.files.length ) {
            showNotification( 'Please select a PDF file.', 'warning' );
            return;
        }

        var file = fileInput.files[0];

        // Client-side validation.
        if ( file.type !== 'application/pdf' && ! file.name.toLowerCase().endsWith( '.pdf' ) ) {
            showNotification( 'Only PDF files are allowed.', 'error' );
            return;
        }

        if ( file.size > 50 * 1024 * 1024 ) {
            showNotification( 'File exceeds the 50 MB size limit.', 'error' );
            return;
        }

        var formData = new FormData();
        formData.append( 'action', 'evil_ai_upload_paper' );
        formData.append( 'nonce', evil_ai_admin.nonce );
        formData.append( 'paper', file );

        var $btn     = $( '#evil-ai-upload-btn' );
        var $spinner = $( '#evil-ai-upload-spinner' );

        $btn.prop( 'disabled', true );
        $spinner.addClass( 'is-active' );

        $.ajax( {
            url: evil_ai_admin.ajax_url,
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            timeout: 120000,
            success: function( response ) {
                if ( response.success ) {
                    showNotification( 'Paper uploaded successfully: ' + escapeHtml( file.name ), 'success' );
                    fileInput.value = '';
                    loadPapers();
                } else {
                    showNotification( 'Upload failed: ' + escapeHtml( response.data.message || 'Unknown error' ), 'error' );
                }
            },
            error: function( xhr ) {
                var msg = 'Upload request failed.';
                if ( xhr.responseJSON && xhr.responseJSON.data && xhr.responseJSON.data.message ) {
                    msg = xhr.responseJSON.data.message;
                }
                showNotification( 'Upload failed: ' + escapeHtml( msg ), 'error' );
            },
            complete: function() {
                $btn.prop( 'disabled', false );
                $spinner.removeClass( 'is-active' );
            }
        } );
    } );

    // -------------------------------------------------------------------------
    // Papers: Delete
    // -------------------------------------------------------------------------

    $( document ).on( 'click', '.evil-ai-delete-btn', function() {
        var filename = $( this ).data( 'filename' );

        if ( ! filename ) {
            return;
        }

        if ( ! confirm( 'Delete "' + filename + '" and all its indexed chunks? This cannot be undone.' ) ) {
            return;
        }

        var $btn = $( this );
        $btn.prop( 'disabled', true );

        $.post( evil_ai_admin.ajax_url, {
            action: 'evil_ai_delete_paper',
            nonce: evil_ai_admin.nonce,
            filename: filename
        }, function( response ) {
            if ( response.success ) {
                showNotification( 'Paper deleted: ' + escapeHtml( filename ), 'success' );
                loadPapers();
            } else {
                showNotification( 'Delete failed: ' + escapeHtml( response.data.message || 'Unknown error' ), 'error' );
                $btn.prop( 'disabled', false );
            }
        } ).fail( function() {
            showNotification( 'Delete request failed. Please try again.', 'error' );
            $btn.prop( 'disabled', false );
        } );
    } );

    // -------------------------------------------------------------------------
    // Papers: Reindex
    // -------------------------------------------------------------------------

    $( '#evil-ai-reindex-btn' ).on( 'click', function() {
        if ( ! confirm( 'Re-index all papers? This may take several minutes depending on the number of papers.' ) ) {
            return;
        }

        var $btn     = $( this );
        var $spinner = $( '#evil-ai-action-spinner' );

        $btn.prop( 'disabled', true );
        $spinner.addClass( 'is-active' );

        showNotification( 'Reindexing started. This may take a while...', 'info' );

        $.post( evil_ai_admin.ajax_url, {
            action: 'evil_ai_reindex',
            nonce: evil_ai_admin.nonce
        }, function( response ) {
            if ( response.success ) {
                showNotification( 'Reindex completed successfully.', 'success' );
                loadPapers();
            } else {
                showNotification( 'Reindex failed: ' + escapeHtml( response.data.message || 'Unknown error' ), 'error' );
            }
        } ).fail( function() {
            showNotification( 'Reindex request failed. The server may have timed out. Check the API logs.', 'error' );
        } ).always( function() {
            $btn.prop( 'disabled', false );
            $spinner.removeClass( 'is-active' );
        } );
    } );

    // -------------------------------------------------------------------------
    // Papers: Refresh
    // -------------------------------------------------------------------------

    $( '#evil-ai-refresh-btn' ).on( 'click', function() {
        loadPapers();
    } );

    // -------------------------------------------------------------------------
    // Settings: Test Connection
    // -------------------------------------------------------------------------

    $( '#evil-ai-test-connection' ).on( 'click', function() {
        var $btn     = $( this );
        var $spinner = $( '#evil-ai-test-spinner' );
        var $result  = $( '#evil-ai-test-result' );

        $btn.prop( 'disabled', true );
        $spinner.addClass( 'is-active' );
        $result.html( '' );

        $.post( evil_ai_admin.ajax_url, {
            action: 'evil_ai_test_connection',
            nonce: evil_ai_admin.nonce
        }, function( response ) {
            if ( response.success ) {
                $result.html(
                    '<p class="evil-ai-test-success">Connection successful!</p>' +
                    '<pre>' + escapeHtml( JSON.stringify( response.data, null, 2 ) ) + '</pre>'
                );
            } else {
                $result.html(
                    '<p class="evil-ai-test-error">Connection failed: ' +
                    escapeHtml( response.data.message || 'Unknown error' ) + '</p>'
                );
            }
        } ).fail( function() {
            $result.html(
                '<p class="evil-ai-test-error">Request failed. Please check the API URL and try again.</p>'
            );
        } ).always( function() {
            $btn.prop( 'disabled', false );
            $spinner.removeClass( 'is-active' );
        } );
    } );

    // -------------------------------------------------------------------------
    // Analytics
    // -------------------------------------------------------------------------

    $( '#evil-ai-load-analytics-btn' ).on( 'click', function() {
        loadAnalytics();
    } );

    /**
     * Load analytics data from the API.
     *
     * @param {number} [days] Number of days (defaults to dropdown value).
     */
    function loadAnalytics( days ) {
        days = days || parseInt( $( '#evil-ai-analytics-days' ).val(), 10 ) || 30;

        var $spinner = $( '#evil-ai-analytics-spinner' );
        $spinner.addClass( 'is-active' );

        $.post( evil_ai_admin.ajax_url, {
            action: 'evil_ai_get_analytics',
            nonce: evil_ai_admin.nonce,
            days: days
        }, function( response ) {
            if ( response.success ) {
                renderAnalytics( response.data );
            } else {
                showNotification(
                    'Failed to load analytics: ' + escapeHtml( response.data.message || 'Unknown error' ),
                    'error'
                );
            }
        } ).fail( function() {
            showNotification( 'Analytics request failed.', 'error' );
        } ).always( function() {
            $spinner.removeClass( 'is-active' );
        } );
    }

    /**
     * Render analytics data into the page.
     *
     * @param {Object} data Analytics response from the API.
     */
    function renderAnalytics( data ) {
        // Summary cards.
        $( '#evil-ai-total-questions' ).text( data.total_questions || data.totalQuestions || 0 );
        $( '#evil-ai-unique-sessions' ).text( data.unique_sessions || data.uniqueSessions || 0 );

        var avgTime = data.avg_response_time || data.avgResponseTime || 0;
        if ( typeof avgTime === 'number' ) {
            avgTime = avgTime.toFixed( 2 ) + 's';
        }
        $( '#evil-ai-avg-response-time' ).text( avgTime );

        var errorRate = data.error_rate || data.errorRate || 0;
        if ( typeof errorRate === 'number' ) {
            errorRate = ( errorRate * 100 ).toFixed( 1 ) + '%';
        }
        $( '#evil-ai-error-rate' ).text( errorRate );

        // Top cited papers.
        var topPapers = data.top_cited_papers || data.topCitedPapers || data.top_papers || [];
        renderBarChart( $( '#evil-ai-top-papers' ), topPapers, 'paper', 'citations', false );

        // Questions per day.
        var perDay = data.questions_per_day || data.questionsPerDay || data.daily || [];
        renderBarChart( $( '#evil-ai-questions-per-day' ), perDay, 'date', 'count', true );

        // Recent questions.
        renderRecentQuestions( data.recent_questions || data.recentQuestions || data.recent || [] );
    }

    /**
     * Render a simple horizontal bar chart.
     *
     * @param {jQuery}  $container The container element.
     * @param {Array}   items      Array of objects.
     * @param {string}  labelKey   Key to use for labels.
     * @param {string}  valueKey   Key to use for values.
     * @param {boolean} isDaily    True for the daily chart variant.
     */
    function renderBarChart( $container, items, labelKey, valueKey, isDaily ) {
        $container.empty();

        if ( ! items || ! items.length ) {
            $container.html( '<p class="evil-ai-placeholder">No data available.</p>' );
            return;
        }

        if ( isDaily ) {
            $container.addClass( 'evil-ai-bar-chart-daily' );
        }

        // Find the max value for scaling.
        var maxVal = 0;
        $.each( items, function( _, item ) {
            var v = item[ valueKey ] || item.value || item.count || 0;
            if ( v > maxVal ) {
                maxVal = v;
            }
        } );

        if ( maxVal === 0 ) {
            maxVal = 1;
        }

        $.each( items, function( _, item ) {
            var label = item[ labelKey ] || item.label || item.name || '';
            var value = item[ valueKey ] || item.value || item.count || 0;
            var pct   = Math.round( ( value / maxVal ) * 100 );

            var row = '<div class="evil-ai-bar-row">' +
                '<span class="evil-ai-bar-label" title="' + escapeAttr( label ) + '">' + escapeHtml( label ) + '</span>' +
                '<div class="evil-ai-bar-track"><div class="evil-ai-bar-fill" style="width:' + pct + '%;"></div></div>' +
                '<span class="evil-ai-bar-value">' + value + '</span>' +
                '</div>';

            $container.append( row );
        } );
    }

    /**
     * Render the recent questions table.
     *
     * @param {Array} questions Array of question objects.
     */
    function renderRecentQuestions( questions ) {
        var $tbody = $( '#evil-ai-recent-questions-tbody' );

        if ( ! questions || ! questions.length ) {
            $tbody.html(
                '<tr><td colspan="4" class="evil-ai-placeholder">No recent questions found.</td></tr>'
            );
            return;
        }

        var rows = '';
        $.each( questions, function( _, q ) {
            var timestamp = q.timestamp || q.created_at || q.date || '';
            var question  = q.question || q.query || q.text || '';
            var papers    = q.papers_cited || q.papersCited || q.sources || [];
            var latency   = q.latency || q.response_time || q.responseTime || '';

            if ( typeof papers === 'object' && Array.isArray( papers ) ) {
                papers = papers.join( ', ' );
            }

            if ( typeof latency === 'number' ) {
                latency = latency.toFixed( 2 ) + 's';
            }

            // Format timestamp for readability.
            if ( timestamp ) {
                try {
                    var d = new Date( timestamp );
                    timestamp = d.toLocaleString();
                } catch ( e ) {
                    // Keep as-is.
                }
            }

            rows += '<tr>';
            rows += '<td>' + escapeHtml( timestamp ) + '</td>';
            rows += '<td>' + escapeHtml( question ) + '</td>';
            rows += '<td>' + escapeHtml( papers ) + '</td>';
            rows += '<td style="text-align:right;">' + escapeHtml( String( latency ) ) + '</td>';
            rows += '</tr>';
        } );

        $tbody.html( rows );
    }

    // -------------------------------------------------------------------------
    // Notification Helper
    // -------------------------------------------------------------------------

    /**
     * Show a notification banner at the top of the page.
     *
     * @param {string} message The message to display.
     * @param {string} type    One of: success, error, info, warning.
     */
    function showNotification( message, type ) {
        type = type || 'info';

        var $container = $( '#evil-ai-notifications' );
        if ( ! $container.length ) {
            return;
        }

        var html = '<div class="evil-ai-notice evil-ai-notice-' + type + '">' +
            '<button type="button" class="evil-ai-notice-dismiss" aria-label="Dismiss">&times;</button>' +
            '<p>' + message + '</p>' +
            '</div>';

        var $notice = $( html );
        $container.append( $notice );

        // Dismiss handler.
        $notice.find( '.evil-ai-notice-dismiss' ).on( 'click', function() {
            $notice.fadeOut( 200, function() {
                $notice.remove();
            } );
        } );

        // Auto-dismiss after 10 seconds for success/info.
        if ( type === 'success' || type === 'info' ) {
            setTimeout( function() {
                $notice.fadeOut( 400, function() {
                    $notice.remove();
                } );
            }, 10000 );
        }
    }

    // -------------------------------------------------------------------------
    // Utility Functions
    // -------------------------------------------------------------------------

    /**
     * Escape a string for safe insertion into HTML.
     *
     * @param {string} str The string to escape.
     * @return {string} Escaped string.
     */
    function escapeHtml( str ) {
        if ( typeof str !== 'string' ) {
            return '';
        }
        var div = document.createElement( 'div' );
        div.appendChild( document.createTextNode( str ) );
        return div.innerHTML;
    }

    /**
     * Escape a string for use in an HTML attribute.
     *
     * @param {string} str The string to escape.
     * @return {string} Escaped string.
     */
    function escapeAttr( str ) {
        if ( typeof str !== 'string' ) {
            return '';
        }
        return str
            .replace( /&/g, '&amp;' )
            .replace( /"/g, '&quot;' )
            .replace( /'/g, '&#39;' )
            .replace( /</g, '&lt;' )
            .replace( />/g, '&gt;' );
    }

    // -------------------------------------------------------------------------
    // Initialize
    // -------------------------------------------------------------------------

    // Load papers on the documents page if we are on that page.
    if ( $( '#evil-ai-papers-table' ).length ) {
        loadPapers();
    }
} );
