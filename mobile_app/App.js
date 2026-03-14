import React, { useRef, useState } from 'react';
import { StyleSheet, View, SafeAreaView, TouchableOpacity, Text, Alert } from 'react-native';
import { WebView } from 'react-native-webview';
import { StatusBar } from 'expo-status-bar';

// Desktop User Agent to ensure we get the same interface as the Python script
const DESKTOP_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36";

export default function App() {
  const webViewRef = useRef(null);
  const [currentUrl, setCurrentUrl] = useState('');

  // Script to execute for purchasing
  const runAutoPurchase = () => {
    const purchaseScript = `
      (function() {
        try {
          // Helper to find element in Document or Iframe
          function getEl(id) {
            var el = document.getElementById(id);
            if (el) return el;
            
            // Try looking inside iframe 'ifrm_tab'
            var iframe = document.getElementById('ifrm_tab');
            if (iframe) {
              try {
                return iframe.contentWindow.document.getElementById(id);
              } catch(e) {
                console.log("Cross-origin iframe access blocked");
              }
            }
            return null;
          }

          // 1. Select 'Automatic Number' Tab
          var tab = getEl('num2');
          if (!tab) {
             // Fallback: Try specific text search or other selectors if needed
             alert('Could not find Auto Tab (num2). Are you on the Game Page?'); 
             return; 
          }
          tab.click();

          // 2. Select Quantity 5
          var sel = getEl('amoundApply');
          if (sel) {
            sel.value = '5';
            // Trigger change event
            if(typeof sel.onchange === 'function') sel.onchange(); 
            // Also try specific function
            try { 
                var iframe = document.getElementById('ifrm_tab');
                if(iframe) iframe.contentWindow.paperTextChange(5);
                else paperTextChange(5);
            } catch(e) {}
          } else {
             alert('Could not find Quantity Selector (amoundApply).');
             return;
          }

          // 3. Click Select (Confirm)
          var btnSelect = getEl('btnSelectNum');
          if (btnSelect) btnSelect.click();
          else { alert('Could not find Select Button'); return; }

          // 4. Click Buy (after small delay)
          setTimeout(function() {
             var btnBuy = getEl('btnBuy');
             if (btnBuy) btnBuy.click();
             
             // 5. Handle Confirm Popup
             setTimeout(function() {
                try {
                   var iframe = document.getElementById('ifrm_tab');
                   if(iframe) iframe.contentWindow.closepopupLayerConfirm(true);
                   else closepopupLayerConfirm(true);
                } catch(e) {
                   // Fallback: click confirm button if visible
                   // var confirmBtn = ...
                }
             }, 1000);
          }, 1000);

        } catch (e) {
          alert('Error: ' + e.message);
        }
      })();
    `;

    webViewRef.current.injectJavaScript(purchaseScript);
  };

  const goToGamePage = () => {
    // Direct link to the game popup URL
    const gameUrl = 'https://el.dhlottery.co.kr/game/TotalGame.jsp?LottoId=LO40';
    webViewRef.current.injectJavaScript(`window.location.href = '${gameUrl}';`);
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="dark" />

      {/* WebView Container */}
      <View style={styles.webviewContainer}>
        <WebView
          ref={webViewRef}
          source={{ uri: 'https://dhlottery.co.kr/common.do?method=main' }}
          userAgent={DESKTOP_USER_AGENT}
          javaScriptEnabled={true}
          domStorageEnabled={true}
          onNavigationStateChange={(navState) => setCurrentUrl(navState.url)}
          style={{ flex: 1 }}
        />
      </View>

      {/* Control Panel */}
      <View style={styles.controls}>
        <TouchableOpacity style={[styles.button, styles.loginBtn]} onPress={() => webViewRef.current.goBack()}>
          <Text style={styles.btnText}>Back</Text>
        </TouchableOpacity>

        <TouchableOpacity style={[styles.button, styles.gameBtn]} onPress={goToGamePage}>
          <Text style={styles.btnText}>Go to Game</Text>
        </TouchableOpacity>

        <TouchableOpacity style={[styles.button, styles.buyBtn]} onPress={runAutoPurchase}>
          <Text style={styles.btnText}>Auto Buy 5</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  webviewContainer: {
    flex: 1,
  },
  controls: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    padding: 15,
    backgroundColor: 'white',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  button: {
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 8,
    minWidth: 80,
    alignItems: 'center',
  },
  loginBtn: {
    backgroundColor: '#9ca3af',
  },
  gameBtn: {
    backgroundColor: '#3b82f6',
  },
  buyBtn: {
    backgroundColor: '#ef4444',
  },
  btnText: {
    color: 'white',
    fontWeight: 'bold',
  }
});
