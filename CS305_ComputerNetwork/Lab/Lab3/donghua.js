$(function(){
	
	
	var  clientheight = $(window).height();
	
	
	
	$(".body_b").scroll(function(){
		var _this = $(this).scrollTop();
		var _bar = $("#focus")[0].getBoundingClientRect().top;
		var _Campus_life = $(".Campus_life")[0].getBoundingClientRect().top;
		var _event = $("#event")[0].getBoundingClientRect().top;
		var _personage = $("#personage")[0].getBoundingClientRect().top;
		var _nkdhz = $(".nkd-hz")[0].getBoundingClientRect().top;
		var _tansuo = $(".nkd-tansuo")[0].getBoundingClientRect().top;
		
		//var _foot = $(".footer")[0].getBoundingClientRect().top;
		
		var _official_video = $(".official_video")[0].getBoundingClientRect().top;
		
		
		/*
		var ft_height = $(".pc_h").height();
		$(".body_b").css("padding-bottom", ft_height+"px");
		*/
		
		//console.log(ft_height );
		
		
		if ($("#focus .swiper-slide").length > 0) {
			if( _this > 0 ) { 
				
				$("#focus .co-title , #focus .btngroup").css({"animation": "fadeIn2 .35s ease-in-out forwards"});				
								
				for(var i = 0 ; i< $("#focus .swiper-slide").length ; i++) {
					$("#focus .swiper-slide").eq(i).find(".itemUp").css({"animation": "fadeInUp2 .5s ease-in-out "+ Math.floor( i * 10 + 5) / 100  +"s forwards"});
					
				}			
				
			}
			
			if( _this > 600 ) {
				
				$(".nkd_highlight .co-title").css({"animation": "fadeIn2 .35s ease-in-out forwards"});				
				for(var i = 0 ; i< $("#focus .swiper-slide").length ; i++) {					
					$("#zhuanti .swiper-slide").eq(i).find("a").css({"animation": "fadeInUp2 .5s ease-in-out "+ Math.floor( i * 10 + 5) / 100  +"s forwards"});
				}			
				
			}
			
		}
	
		
		if ($(".official_video").length > 0) {
			if( Math.abs(_official_video - _this) < 200  ) { 				
				$(".official_video").css({"animation": "fadeInUp2 .5s ease-in-out .5s forwards"});
				$(".about_nkd").css({"animation": "fadeInUp2 .5s ease-in-out .75s forwards"});
			}
			
		}
		
		
		if ($("#personage").length > 0) {
			
			if( _personage - clientheight < -230  ) {
				$(".nkd-Academics .co-title").css({"animation": "fadeIn2 .35s ease-in-out forwards"});	
				$("#personage").css({"animation": "fadeInUp2 .5s ease-in-out .35s forwards"});
			}
		}
		
		if ($(".nkd-hz").length > 0) {
			
			if( _nkdhz - clientheight < -230  ) {
				$(".nkd-hz .co-title").css({"animation": "fadeIn2 .35s ease-in-out forwards"});
				$(".new-3").css({"animation": "fadeInUp2 .5s ease-in-out forwards"});
				for(var i = 0 ; i< $(".nkd-hz .edu_enter ul li").length ; i++) {
					$(".nkd-hz .edu_enter ul li").eq(i).css({"animation": "fadeInUp2 .5s ease-in-out "+ Math.floor( i * 10 + 5) / 100  +"s forwards"});
				}
			}
		}
		
		
		if ($(".nkd-tansuo").length > 0) {
			
			var slik_h = $("#research_lunbo .gallery-thumbs .swiper-wrapper").height();
			var _one = slik_h / 4;
			// console.log(slik_h);
			
			$("#research_lunbo .gallery-thumbs .swiper-slide").mouseenter(function(){
				var _index = $(this).index();
				$(this).addClass("spider-color").siblings().removeClass("spider-color");
				$("span.slikboder").css({"transform":"translateY("+ _one * _index +"px)", "transition":" all .35s"});
			});
			
			$("#research_lunbo .gallery-thumbs .swiper-slide").mouseleave(function(){		
				$("#research_lunbo .gallery-thumbs .swiper-slide:nth-child(1)").addClass("spider-color").siblings().removeClass("spider-color");
				$("span.slikboder").css({"transform":"translateY(0px)", "transition":" all .35s"});
			});		
			
			
			if( _tansuo - clientheight < -230 ) {				
				$("#research_lunbo .gallery-top").addClass("imgOpen");
				$(".nkd-tansuo .co-title,div#research_lunbo ").css({"animation": "fadeIn2 .35s ease-in-out forwards"});
				for(var i = 0 ; i< $(".nkd-tansuo .edu_enter ul li").length ; i++) {
					$(".nkd-tansuo .edu_enter ul li").eq(i).css({"animation": "fadeInUp2 .5s ease-in-out "+ Math.floor( i * 10 + 5) / 100  +"s forwards"});
				}
			}
			
			
		}
		
		
		if ($("#home_events .swiper-slide").length > 0) {
			
			if(  _event - clientheight < -300 ) {
				$("#event .co-title, #home_anoucement").css({"animation": "fadeIn2 .35s ease-in-out forwards"});
				for(var i = 0 ; i< $("#home_events .swiper-slide").length ; i++) {
					$("#home_events .swiper-slide ").eq(i).find("dl.event_item ").css({"animation": "fadeInUp2 .5s ease-in-out "+ Math.floor( i * 10 - 10 ) / 100  +"s forwards"});
				}
				
				//$(".footer").show();
			}
			
			
		}
		
		
		if ($(".CampusLife_box ul li").length > 0) {
			
			if(  _Campus_life - clientheight < -400 ) {
				$("#xysh .co-title").css({"animation": "fadeIn2 .35s ease-in-out forwards"});
				for(var i = 0 ; i< $(".CampusLife_box ul li").length ; i++) {
					$(".CampusLife_box ul li.itemUp").eq(i).css({"animation": "fadeInUp2 .5s ease-in-out "+ Math.floor( i * 10 + 5) / 100  +"s forwards"});
				}
			}
		}
		
		
		
		
		
		
	})
})