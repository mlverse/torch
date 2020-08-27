
$(document).ready(function() {
  $('#menuToggler').on('click', function(event) {
    if ($('#menuItems').hasClass('showMenu')) {
      $('#menuItems').removeClass("showMenu");
    }
    else {
      $('#menuItems').addClass("showMenu");
    }
  });
});
