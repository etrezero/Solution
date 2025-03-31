// Split.js 기본 구동
document.addEventListener("DOMContentLoaded", function () {
  Split(['#left-pane', '#right-pane'], {
    sizes: [50, 50],
    minSize: 200,
    gutterSize: 8,
    cursor: 'ew-resize',
  });
});
