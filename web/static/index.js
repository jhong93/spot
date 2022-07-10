const GROUND_TRUTH_LABELS = {};
var PRED_LABELS = null;


function clearView() {
    $('#missingDetDiv').empty();
    $('#closeDetDiv').empty();
    $('#spuriousDetDiv').empty();
}


function showFrames(elem, video, event, true_event, aux_events,
                    aux_true_events) {
    let parent = elem.closest('.event-block');
    var frame_div = parent.find('.frame-div');
    if (frame_div.length == 0) {
        frame_div = $('<div>').addClass('frame-div');

        let aux_frames = new Set();
        if (aux_events) {
            aux_events.forEach(e => aux_frames.add(e.event.frame));
        }

        let aux_true_frames = new Set();
        if (aux_true_events) {
            aux_true_events.forEach(e => aux_true_frames.add(e.frame));
        }

        function get_aux_event(i) {
            let e = aux_events.filter(e => e.event.frame == i)[0].event;
            return `${e.label} (${e.score.toFixed(3)})`;
        }

        function addFrame(i, context_type) {
            let rgb = $('<a>').prop({
                target: '_blank', href: `full_res/${video.replaceAll('/', '=')}/${i}`
            }).append($('<img>').prop('src', `/rgb/${video.replaceAll('/', '=')}/${i}`));
            let flow = $('<img>').prop('src', `/flow/${video.replaceAll('/', '=')}/${i}`);
            frame_div.append(
                $('<div>').addClass('frame-block').append(
                    $('<div>').addClass('frame-info').append(
                        $('<span>').text(i),
                        context_type ?
                            $('<span>').addClass('context').text(
                                `${context_type} context`) : null,
                        true_event && true_event.frame == i ?
                            $('<span>').addClass('ground-truth').text(
                                `ground truth: ${true_event.label}`) : null,
                        event && event.frame == i ?
                            $('<span>').addClass('predicted').html(
                                `prediction:<br>${event.label} (${event.score.toFixed(3)})`) : null,
                        aux_events && aux_frames.has(i) ?
                            $('<span>').addClass('predicted-other').html(
                                `other prediction:<br>${get_aux_event(i)}`) : null,
                        aux_true_events && aux_true_frames.has(i) ?
                            $('<span>').addClass('ground-truth-other').html(
                                `other truth:<br>${
                                    aux_true_events.filter(
                                        e => e.frame == i)[0].label
                                }`) : null,
                    ),
                    context_type == 'sparse' ?
                        rgb.addClass('context-frame') : rgb,
                    context_type == 'sparse' ?
                        flow.addClass('context-frame') : flow,
                )
            )
        }

        let tol = parseInt($('#numFrames').val());
        let sparse_context = parseInt($('#numContextFrames').val());
        let dense_context = tol + sparse_context;
        let context_stride = 5;
        let frame = event == null ? true_event.frame : event.frame;
        for (let i = frame - dense_context - sparse_context * context_stride;
             i < frame - dense_context; i += context_stride) {
            addFrame(i, 'sparse');
        }
        for (let i = frame - dense_context; i <= frame + dense_context; i++) {
            if (i == frame - tol) {
                frame_div.append($('<hr>'));
            }
            addFrame(i, Math.abs(frame - i) <= tol ? null : 'dense');
            if (i == frame + tol) {
                frame_div.append($('<hr>'));
            }
        }
        for (let i = frame + dense_context + context_stride;
             i <= frame + dense_context + sparse_context * context_stride; i += context_stride) {
            addFrame(i, 'sparse');
        }

        frame_div.append($('<span>').addClass('event-info').append(
            $('<a>').addClass('show-frames').text('+/-').click(() => {
                frame_div.remove();
                parent.get(0).scrollIntoView(true);
            })));
        parent.append(frame_div);
    } else {
        frame_div.remove();
        parent.get(0).scrollIntoView(true);
    }
}


function setVideoFilter(v) {
    $('#videoFilter').val(v).trigger("chosen:updated").change();
}


function setLabelFilter(l) {
    $('#labelFilter').val(l).trigger("chosen:updated").change();
}


function renderMissing(missing) {
    let div = $('#missingDetDiv');
    $('#missingDetCount').text(missing.length);
    missing.forEach(e => {
        div.append(
            $('<div>').addClass('event-block').append(
                $('<span>').addClass('event-info').append(
                    $('<a>').addClass('show-frames').text('+/-').click(
                        function() {
                            let truth_other = GROUND_TRUTH_LABELS[e.video
                                ].events.filter(x => e.event.frame != x.frame);
                            showFrames($(this), e.video, null, e.event,
                                       e.pred_events, truth_other);
                        }
                    )),
                $('<span>').addClass('event-info').append(
                    $('<span>').addClass('key').text('ground truth')),
                $('<span>').addClass('event-info').append(
                    $('<span>').addClass('key').text('label'),
                    $('<span>').addClass('value').text(e.event.label)).click(
                        () => setLabelFilter(e.event.label)),
                $('<span>').addClass('event-info').append(
                    $('<span>').addClass('key').text('frame'),
                    $('<span>').addClass('value').text(e.event.frame)),
                $('<span>').addClass('event-info').append(
                    $('<span>').addClass('key').text(e.event.comment)),
                $('<span>').addClass('event-info').append(
                    $('<span>').addClass('key').text(e.video)).click(
                        () => setVideoFilter(e.video))
            ));
    });
}


function sortByScore(a, b) { return b.event.score - a.event.score; }
function sortByVideoFrame(a, b) {
    if (b.video < a.video) {
        return 1;
    } else if (b.video > a.video) {
        return -1;
    } else {
        return a.event.frame - b.event.frame;
    }
}


function renderClose(close) {
    let div = $('#closeDetDiv');
    let gt_found = new Set();
    let events = close.sort(sortByScore).map(e => {
        let k = `${e.video}@${e.true_event.frame}`;
        e.is_duplicate = gt_found.has(k);
        gt_found.add(k);
        return e;
    });

    if ($('#sortBy').val() == 'video') {
        events.sort(sortByVideoFrame);
    }

    let num_duplicates = events.filter(x => x.is_duplicate).length;
    $('#closeDetCount').text(events.length - num_duplicates);
    $('#duplicateDetCount').text(num_duplicates);

    events.forEach(e => {
        div.append(
            $('<div>').addClass('event-block').append(
                $('<span>').addClass('event-info').append(
                    $('<a>').addClass('show-frames').text('+/-').click(
                        function() {
                            showFrames($(this), e.video, e.event, e.true_event, close.filter(
                                x => x.video == e.video
                                && x.event.label == e.event.label
                                && x.event.frame != e.event.frame));
                        }
                    )),
                $('<span>').addClass('event-info').append(
                    $('<span>').addClass('key').text('score'),
                    $('<span>').addClass('value').text(
                        e.event.score.toFixed(3))),
                $('<span>').addClass('event-info').append(
                    $('<span>').addClass('key').text('is dupl'),
                    $('<span>').addClass('value').css({
                        color: e.is_duplicate ? '#f00': null
                    }).text(e.is_duplicate)),
                $('<span>').addClass('event-info').append(
                    $('<span>').addClass('key').text('label'),
                    $('<span>').addClass('value').text(e.event.label)).click(
                        () => setLabelFilter(e.event.label)),
                $('<span>').addClass('event-info').append(
                    $('<span>').addClass('key').text('frame'),
                    $('<span>').addClass('value').text(e.event.frame)),
                $('<span>').addClass('event-info').append(
                    $('<span>').addClass('key').text('ofs from gt'),
                    $('<span>').addClass('value').text(
                        e.event.frame - e.true_event.frame)),
                $('<span>').addClass('event-info').append(
                    $('<span>').addClass('key').text(e.true_event.comment)),
                $('<span>').addClass('event-info').append(
                    $('<span>').addClass('key').text(e.video)).click(
                        () => setVideoFilter(e.video))
            ));
    });
}


function renderSpurious(spurious) {
    let div = $('#spuriousDetDiv');
    let order_by = $('#sortBy').val() == 'score' ?
        sortByScore : sortByVideoFrame;

    $('#spuriousDetCount').text(spurious.length);
    spurious.sort(order_by).forEach(e => {
        let truth_other = GROUND_TRUTH_LABELS[e.video].events.filter(
            x => e.event.label != x.label);
        div.append(
            $('<div>').addClass('event-block').append(
                $('<span>').addClass('event-info').append(
                    $('<a>').addClass('show-frames').text('+/-').click(
                        function() {
                            showFrames($(this), e.video, e.event, e.true_event,
                                spurious.filter(x =>
                                    x.video == e.video
                                    // && x.event.label == e.event.label
                                    && x.event.frame != e.event.frame),
                                truth_other
                            );
                        }
                    )),
                $('<span>').addClass('event-info').append(
                    $('<span>').addClass('key').text('score'),
                    $('<span>').addClass('value').text(
                        e.event.score.toFixed(3))),
                $('<span>').addClass('event-info').append(
                    $('<span>').addClass('key').text('label'),
                    $('<span>').addClass('value').text(e.event.label)).click(
                        () => setLabelFilter(e.event.label)),
                $('<span>').addClass('event-info').append(
                    $('<span>').addClass('key').text('frame'),
                    $('<span>').addClass('value').text(e.event.frame)),
                $('<span>').addClass('event-info').append(
                    $('<span>').addClass('key').text(e.video)).click(
                        () => setVideoFilter(e.video))
            ));
    });
}


function refreshView() {
    clearView();
    let tol = $('#numFrames').val();
    let label_filter = $('#labelFilter').val();
    let video_filter = $('#videoFilter').val();
    if (video_filter) {
        $('.result-div').show();
    }

    let missing = [];
    let close = [];
    let spurious = [];

    PRED_LABELS.forEach(pred => {
        if (video_filter && pred.video != video_filter) {
            return;
        }

        let truth = GROUND_TRUTH_LABELS[pred.video];
        let gt_found = new Set();
        pred.events.forEach(e1 => {
            if (label_filter && e1.label != label_filter) {
                return;
            }

            let gt_matches = truth.events.filter(e2 =>
                e1.label == e2.label && Math.abs(e1.frame - e2.frame) <= tol);
            let record = {'video': pred.video, 'event': e1};
            if (gt_matches.length == 0) {
                record.true_event = truth.events.filter(
                    e2 => e1.label == e2.label).reduce(
                        (a, b) => !a ||
                            Math.abs(e1.frame - b.frame)
                            < Math.abs(e1.frame - a.frame) ? b : a, undefined);
                spurious.push(record);
            } else {
                record.true_event = gt_matches[0];
                close.push(record);
                if (gt_matches.length != 1) {
                    console.log('Awk: prediction matched multiple');
                }
            }
            gt_matches.forEach(e => gt_found.add(e.frame));
        });

        truth.events.filter(e =>
            !gt_found.has(e.frame) && (!label_filter || e.label == label_filter)
        ).forEach(e => missing.push({
            video: truth.video, event: e,
            pred_events: pred.events.map(
                x => ({video: truth.video, event: x}))
        }));
    });

    renderMissing(missing);
    renderClose(close);
    renderSpurious(spurious);
}


function initialize() {
    $('#predName').chosen({search_contains: true}).change(() => {
        let pred_name = $('#predName').val();
        $.get(`/pred/${pred_name}`, function(data) {
            PRED_LABELS = data;

            let pred_videos = new Set();
            PRED_LABELS.forEach(e => pred_videos.add(e.video));
            $('#videoFilter').find('option').each(function() {
                $(this).prop(
                    'disabled',
                    $(this).val() == '' || pred_videos.has($(this).val())
                    ? 0 : 1);
            });
            $('#videoFilter').trigger('chosen:updated');

            if ($('#autoRefresh').is(":checked")) {
                refreshView();
            }
        });
    });
    $('#numFrames').change(refreshView);
    $('#labelFilter').chosen({search_contains: true}).change(refreshView);
    $('#videoFilter').chosen({search_contains: true}).change(refreshView);
    $('#sortBy').chosen({search_contains: true}).change(refreshView);
    $('.toggle-result').click(function(e) {
        $(this).closest('.content-div').find('.result-div').toggle();
    });

    // Init
    $('#predName').change();
}


$.get('labels.json', function(data) {
    data.forEach(e => {
        GROUND_TRUTH_LABELS[e.video] = e;
    });
    initialize();
});