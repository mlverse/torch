
$(document).ready(function() {
  $(window).on('scroll', function(event) {
    var scrollPos = $(event.target).scrollTop();
    if (scrollPos > 20) {
      $('#appTidyverseSite').addClass("shrinkHeader");
    }
    else {
      $('#appTidyverseSite').removeClass("shrinkHeader");
    }
  });

});
