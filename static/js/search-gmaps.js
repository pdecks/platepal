//TODO: document ready issue
// define arrays for the markers by category
var markers = new Array();

var catCodes = ['gltn', 'vgan', 'kshr', 'algy', 'pleo', 'unkn'];
var catNames = {'gltn': 'Gluten-Free',
                'vgan': 'Vegan',
                'kshr': 'Kosher',
                'algy': 'Allergies',
                'pleo': 'Paleo',
                'unkn': 'Feeling Lucky'
                };

// for keeping track of results loaded on page
var catCounts = {'gltn': 0,
                 'vgan': 0,
                 'kshr': 0,
                 'algy': 0,
                 'pleo': 0,
                 'unkn': 0};

var styles = [
    {
        "featureType": "administrative",
        "elementType": "all",
        "stylers": [
            {
                "visibility": "on"
            },
            {
                "lightness": 33
            }
        ]
    },
    {
        "featureType": "landscape",
        "elementType": "all",
        "stylers": [
            {
                "color": "#f2e5d4"
            }
        ]
    },
    {
        "featureType": "poi.park",
        "elementType": "geometry",
        "stylers": [
            {
                "color": "#c5dac6"
            }
        ]
    },
    {
        "featureType": "poi.park",
        "elementType": "labels",
        "stylers": [
            {
                "visibility": "on"
            },
            {
                "lightness": 20
            }
        ]
    },
    {
        "featureType": "road",
        "elementType": "all",
        "stylers": [
            {
                "lightness": 20
            }
        ]
    },
    {
        "featureType": "road.highway",
        "elementType": "geometry",
        "stylers": [
            {
                "color": "#c5c6c6"
            }
        ]
    },
    {
        "featureType": "road.arterial",
        "elementType": "geometry",
        "stylers": [
            {
                "color": "#e4d7c6"
            }
        ]
    },
    {
        "featureType": "road.local",
        "elementType": "geometry",
        "stylers": [
            {
                "color": "#fbfaf7"
            }
        ]
    },
    {
        "featureType": "water",
        "elementType": "all",
        "stylers": [
            {
                "visibility": "on"
            },
            {
                "color": "#acbcc9"
            }
        ]
    }
]


function getCityState(){
  var city_name = $("#search-city").text();
  var state = $("#search-state").text();
  console.log(city_name);
  console.log(state);
  return [city_name, state];

}


function initMap(){
  console.log("initMap called");
  // get city, state
  var location = getCityState();
  var city = location[0];
  var state = location[1];
  var address = city + ', ' + state;
  console.log(city);
  console.log(state);
  console.log(address);

  pageGeo = "/" + state + "/" + city + "/geocode.json";
  $.getJSON(pageGeo, function(geocodeJSON){
  // var myLatLng = {lat: 37.435, lng: -122.17};
    // define map
    var map = new google.maps.Map(document.getElementById('map'),{
      center: geocodeJSON,
      // center: myLatLng,
      zoom: 11,

    });
    map.setOptions({styles: styles});
    // define markers
    setMarkers(map);

  }); // end $.get(geocodeJSON)
  
} // end initMap


// JSON = {'gltn': [{'biz_id': biz_id, 'avg_cat_review': avg_cat_review, 'lat': lat, 'lng': lng}, {}, {}],
//         'vgan': [{}, {}, {}],
//          ...
//         }

function getSearchTerms() {
  var term;
  var searchString = "";
  var searchTerms = [];
  $(".search-terms").each(function(){
    term = $(this).text();
    searchString += term + "%20";
    searchTerms.push(term);
  });

  return [searchString, searchTerms];
}

function setMarkers(map) {
  console.log("setMarkers called");
  // Adds markers to the map.
  
  locationArray = getCityState();
  // city = locationArray[0].split('%20').join(' ');
  // state = locationArray[1];

  // '/<state>/<city>/city.json'
  searchInfo = getSearchTerms();
  searchString = searchInfo[0];  // string separated by %20
  searchTerms = searchInfo[1];  // array of search terms

  console.log(searchInfo);
  console.log(searchString);
  console.log(locationArray);
  
  var searchPage='/'+searchString+"/"+ locationArray[0]+'/'+locationArray[1]+'/search.json';
  console.log(searchPage);

  $.getJSON(searchPage, function(top5json) {
      console.log('in $.get top5json');
      console.log(top5json);
      
      // iterate over each item in the dictionary
      for (var cat in top5json) {
        markers[cat] = new Array();

        var resList = $("#results-list-"+cat);
        // console.log('This is resList');
        // console.log(resList);
        var infoWindow = new google.maps.InfoWindow();

        for (var i=0; i<top5json[cat].length; i++){
          var biz = top5json[cat][i];
          var letter = String.fromCharCode("A".charCodeAt(0) + i);
          var latLng = new google.maps.LatLng(biz.lat, biz.lng);

          var marker = new google.maps.Marker({
            position: latLng,
            title: biz.name,
            icon: "http://maps.google.com/mapfiles/marker" + letter + ".png"
          });
          
          marker.setMap(map);

          if (cat !== 'gltn'){
            marker.setVisible(false);
          }

          // create an event handler to listen for marker clicks
          // opens an infoWindow on the marker when clicked
          (function (marker, biz) {
            google.maps.event.addListener(marker, "click", function (e){
              //wrap the content inside an html div to set height and width of InfoWindow
              infoWindow.setContent('<div id="content" style="width:200px;min-height:40px">'+
                '<div id="siteNotice">'+
                '</div>'+
                '<h3 id="firstHeading" class="firstHeading">'+ biz.name + '</h3>'+
                '<div id="bodyContent">'+
                'Average Review by Category: ' + biz.avg_cat_review + '</br>' +
                '</div>'+
                '</div>');
              // infoWindow.setPosition()
              infoWindow.open(map, marker);
            });
          }) (marker, biz); // TODO: understand this

          markers[cat].push(marker);

          // var avg_review;
          // if (biz.avg_cat_review !== 'undefined'){
          //   avg_cat_review = 'N/A';
          // }
          // else {
          //   avg_review = biz.avg_cat_review;
          // }
          resList.append("<li><a href='/biz/"+biz.biz_id+"'>" + biz.name + "</a><br>PlatePal Score: " + biz.avg_cat_review + "</br></li>");
          console.log('appended to resList');
        } // end inner for loop over businesses in list by category
                
      } // end outer for loop over categories
      
  }); // end $.getJSON

} // end setMarkers


function filterResults(cat) {
  // slice off the 4 letter cat code
  
  // turn on markers in current cat
  // show list for current cat
  for (var j in markers[cat]){
    markers[cat][j].setVisible(true);
  }
  var name = "div#results-"+cat;
  $(name).removeClass("hidden");
  // styleAttr = $("div.results-"+cat).attr("style");
  // console.log('this is stleAttr for ' + cat);
  console.log('"'+'div#results-'+cat+'"');
  // console.log(styleAttr);

  // turn off markers in other cats
  // hide lists for other cats
  for (var i = 0; i < catCodes.length; i++){
    var currentCat = catCodes[i];

    if (currentCat !== cat){
      name = "div#results-"+currentCat;
      $(name).addClass("hidden");
      for (var k in markers[currentCat]){
        markers[currentCat][k].setVisible(false);
      } // end inner for
    } // end if
  } // end for 
} // end filterResults

// TODO: make this a thing
// function addMarkersByCat(cat, page){
// // Adds markers to the map.
//   // page = "/popular-biz.json/5/0"
//   console.log("this is page in addMarkersByCat");
//   console.log(page);
//   $.get(page, function(nextJSON) {
//       console.log('in $.get');
//       console.log(nextJSON);
//       // iterate over each item in the dictionary with $.each()
//       for (var cat in nextJSON) {
//         // markers[cat] = new Array(); --> already exists from initMap

//         var resList = $("#results-list-"+cat);
//         var infoWindow = new google.maps.InfoWindow();

//         var j = 0;
//         for (var i=markers[cat].length; i<(markers[cat].length + nextJSON[cat].length); i++){
//           var biz = nextJSON[cat][i];
//           var letter = String.fromCharCode("A".charCodeAt(0) + j);
//           var latLng = new google.maps.LatLng(biz.lat, biz.lng);

//           var marker = new google.maps.Marker({
//             position: latLng,
//             title: biz.name,
//             icon: "http://maps.google.com/mapfiles/marker" + letter + ".png"
//           });
          
//           marker.setMap(map);

//           // create an event handler to listen for marker clicks
//           // opens an infoWindow on the marker when clicked
//           (function (marker, biz) {
//             google.maps.event.addListener(marker, "click", function (e){
//               //wrap the content inside an html div to set height and width of InfoWindow
//               infoWindow.setContent('<div id="content" style="width:200px;min-height:40px">'+
//                 '<div id="siteNotice">'+
//                 '</div>'+
//                 '<h3 id="firstHeading" class="firstHeading">'+ biz.name + '</h3>'+
//                 '<div id="bodyContent">'+
//                 'Average Review by Category: ' + biz.avg_cat_review + '</br>' +
//                 '</div>'+
//                 '</div>');
//               // infoWindow.setPosition()
//               infoWindow.open(map, marker);
//             });
//           }) (marker, biz); // TODO: understand this

//           markers[cat].push(marker);

//           resList.append("<li>" + biz.name + " " + biz.avg_cat_review + "</li>");
//         } // end inner for loop over businesses in list by category
                
//       } // end outer for loop over categories
      
//   }); // end $.getJSON

// } // end addMarkersByCat

// TODO: make this a thing
// // each time AJAX is used to update the results, update the dictionary
// function updateCatCounts(cat) {
  

// }



// create click listener on all links in map navigation bar
$("a.map-nav").on('click', function(evt){
  // get the link's id
  var catID = $(this).attr('id');
  var cat = catID.slice(0,4);
  // filter the map results based on the category
  filterResults(cat);

});


// $(document).ready(initMap());
// initMap();
// create click listener on all update results links results list
// $("a.update-results").on('click', function(evt){
// $("a#gltn-next-5.update-results").on('click', function(evt){

//   // make AJAX call to update results
//   console.log('i was clicked');
//   evt.preventDefault();
//   alert("what what");
//   var catID = $(this).attr('id');
//   var page = $(this).attr('href');
//   console.log(page);
//   var cat = catID(0,4);
//   // update the displayed results
//   addMarkersByCat(cat, page);

//   // update the Cat Counts and the href link after JSON returned (may be fewer than 5 results)

// });



